//! Cœur Rust de GPUsync (sans Python) : frontend + génération de code + exécution sur un backend.
//!
//! - [`parse`] : AST Python (JSON) → IR typée ;
//! - [`Device`] : périphérique ouvert (`opencl:0`, `metal:0`, `vulkan:1`, `dx12:0`, `cpu:0`) ;
//! - [`compile`] : vérifie les capacités, génère et compile le code (avec cache) ;
//! - [`launch`] : exécute sur `n` éléments, en blocs, selon le mode choisi (moteur de transferts,
//!   ARCHITECTURE.md §9) et renvoie un [`DeviceArray`].

use std::sync::Arc;

use gpusync_backend_opencl::{ClDevice, ClKernel};
use gpusync_backend_wgpu::{WgDevice, WgKernel};
use gpusync_ir::Kernel;
use gpusync_runtime::{DeviceInfo, DeviceKind};

pub use gpusync_frontend::{ErrorKind as FrontendErrorKind, FrontendError};

/// Erreurs, chacune associée à une exception Python (§4.10).
#[derive(Debug, Clone, PartialEq)]
pub enum CoreError {
    Syntax(String),
    Type(String),
    Shape(String),
    Unsupported(String),
    Device(String),
}

impl std::fmt::Display for CoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (CoreError::Syntax(m)
        | CoreError::Type(m)
        | CoreError::Shape(m)
        | CoreError::Unsupported(m)
        | CoreError::Device(m)) = self;
        f.write_str(m)
    }
}

impl std::error::Error for CoreError {}

type R<T> = Result<T, CoreError>;

fn dev_err(e: impl std::fmt::Display) -> CoreError {
    CoreError::Device(e.to_string())
}

/// AST Python (JSON, voir gpusync-frontend) → IR typée.
pub fn parse(spec_json: &str) -> R<Arc<Kernel>> {
    gpusync_frontend::lower_json(spec_json)
        .map(Arc::new)
        .map_err(|e| match e.kind {
            FrontendErrorKind::Syntax => CoreError::Syntax(e.to_string()),
            FrontendErrorKind::Type => CoreError::Type(e.to_string()),
        })
}

/// Code généré pour une cible : "opencl", "wgsl" ou "ir".
pub fn source(k: &Kernel, target: &str) -> R<String> {
    match target {
        "opencl" => gpusync_codegen::emit_opencl(k)
            .map(|s| s.code)
            .map_err(|e| CoreError::Unsupported(e.0)),
        "wgsl" | "metal" | "vulkan" | "dx12" => gpusync_codegen::emit_wgsl(k)
            .map(|s| s.code)
            .map_err(|e| CoreError::Unsupported(e.0)),
        "ir" => Ok(format!("{k:#?}")),
        t => Err(CoreError::Type(format!(
            "cible inconnue « {t} » (opencl, wgsl ou ir)"
        ))),
    }
}

/// Description de l'interpréteur, exposé comme le périphérique `cpu:0`.
pub fn cpu_info() -> DeviceInfo {
    DeviceInfo {
        name: "Interpréteur GPUsync (CPU, lent)".into(),
        backend: "cpu",
        index: 0,
        kind: DeviceKind::Cpu,
        vendor: "GPUsync".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        f64: true,
        f64_math: true,
        unified_memory: true,
        memory_bytes: None,
        max_buffer_bytes: None,
    }
}

/// Un périphérique ouvert.
#[derive(Clone)]
pub enum Device {
    Cl(Arc<ClDevice>),
    Wg(Arc<WgDevice>),
    Cpu,
}

impl Device {
    /// Ouvre `opencl:<i>`, `metal:<i>`, `vulkan:<i>`, `dx12:<i>` ou `cpu:0`.
    pub fn open(id: &str) -> R<Device> {
        let (backend, index) = id.split_once(':').unwrap_or((id, "0"));
        let index: usize = index.parse().map_err(|_| {
            CoreError::Device(format!("identifiant de périphérique invalide : {id}"))
        })?;
        match backend {
            "opencl" => ClDevice::open(index).map(Device::Cl).map_err(dev_err),
            "metal" | "vulkan" | "dx12" => WgDevice::open(backend, index)
                .map(Device::Wg)
                .map_err(dev_err),
            "cpu" if index == 0 => Ok(Device::Cpu),
            _ => Err(CoreError::Device(format!("périphérique inconnu : {id}"))),
        }
    }

    pub fn info(&self) -> DeviceInfo {
        match self {
            Device::Cl(d) => d.info.clone(),
            Device::Wg(d) => d.info.clone(),
            Device::Cpu => cpu_info(),
        }
    }

    pub fn id(&self) -> String {
        self.info().id()
    }

    /// Entrées et sortie en mémoire de l'hôte, sans copie ?
    pub fn zero_copy(&self) -> bool {
        match self {
            Device::Cl(d) => d.info.unified_memory,
            Device::Wg(d) => d.info.backend == "metal", // newBufferWithBytesNoCopy
            Device::Cpu => true,
        }
    }

    /// Attend la fin de tous les calculs en file.
    pub fn finish(&self) -> R<()> {
        match self {
            Device::Cl(d) => d.finish().map_err(dev_err),
            Device::Wg(d) => d.finish().map_err(dev_err),
            Device::Cpu => Ok(()),
        }
    }
}

/// Kernel prêt à lancer sur un périphérique donné.
#[derive(Clone)]
pub struct Compiled {
    pub kernel: Arc<Kernel>,
    pub(crate) imp: Imp,
}

#[derive(Clone)]
pub(crate) enum Imp {
    Cl(Arc<ClKernel>),
    Wg(Arc<WgKernel>),
    Cpu,
}

/// Raison pour laquelle `info` ne peut pas exécuter `k` (§6.2), ou `None` s'il le peut.
pub fn unsupported(info: &DeviceInfo, k: &Kernel) -> Option<String> {
    let req = k.requirements();
    if req.f64 && !info.f64 {
        return Some(format!(
            "le kernel '{}' utilise f64, non supporté par '{}' ({})",
            k.name,
            info.name,
            info.id()
        ));
    }
    if let Some(op) = req.f64_math_op.filter(|_| !info.f64_math) {
        return Some(format!(
            "le kernel '{}' utilise {op} en f64, non disponible sur '{}' ({}) : seuls OpenCL, CUDA et cpu:0 \
             fournissent les fonctions mathématiques en double précision",
            k.name,
            info.name,
            info.id()
        ));
    }
    None
}

/// Vérifie les capacités du périphérique (§6.2), génère le code et le compile.
pub fn compile(dev: &Device, k: &Arc<Kernel>) -> R<Compiled> {
    let info = dev.info();
    if let Some(msg) = unsupported(&info, k) {
        return Err(CoreError::Unsupported(msg));
    }
    let imp = match dev {
        Device::Cl(d) => {
            let src = gpusync_codegen::emit_opencl(k).map_err(|e| CoreError::Unsupported(e.0))?;
            Imp::Cl(d.compile(&src.code, &src.entry).map_err(dev_err)?)
        }
        Device::Wg(d) => {
            let src = gpusync_codegen::emit_wgsl(k).map_err(|e| CoreError::Unsupported(e.0))?;
            Imp::Wg(
                d.compile(
                    &src.code,
                    k.array_params().count(),
                    gpusync_codegen::has_f64_params(k),
                )
                .map_err(dev_err)?,
            )
        }
        Device::Cpu => Imp::Cpu,
    };
    Ok(Compiled {
        kernel: k.clone(),
        imp,
    })
}

mod exec;

pub use exec::{
    DeviceArray, Input, Mode, Storage, TransferLimits, launch, set_transfer_limits, transfer_limits,
};
