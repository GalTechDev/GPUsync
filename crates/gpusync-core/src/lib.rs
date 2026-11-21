//! Cœur Rust de GPUsync (sans Python) : frontend + génération de code + exécution sur un backend.
//!
//! - [`parse`] : AST Python (JSON) → IR typée ;
//! - [`Device`] : périphérique ouvert (`cuda:0`, `opencl:0`, `metal:0`, `vulkan:1`, `dx12:0`, `cpu:0`) ;
//! - [`compile`] : vérifie les capacités, génère et compile le code (avec cache) ;
//! - [`launch`] : exécute sur `n` éléments, en blocs, selon le mode choisi (moteur de transferts,
//!   ARCHITECTURE.md §9) et renvoie un [`DeviceArray`].

use std::collections::HashMap;
use std::sync::Arc;

use gpusync_backend_cuda::{CuDevice, CuKernel};
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

/// Code généré pour une cible : "cuda", "opencl", "wgsl" ou "ir".
pub fn source(k: &Kernel, target: &str) -> R<String> {
    match target {
        "cuda" => gpusync_codegen::emit_cuda(k)
            .map(|s| s.code)
            .map_err(|e| CoreError::Unsupported(e.0)),
        "opencl" => gpusync_codegen::emit_opencl(k)
            .map(|s| s.code)
            .map_err(|e| CoreError::Unsupported(e.0)),
        "wgsl" | "metal" | "vulkan" | "dx12" => gpusync_codegen::emit_wgsl(k)
            .map(|s| s.code)
            .map_err(|e| CoreError::Unsupported(e.0)),
        "ir" => Ok(format!("{k:#?}")),
        t => Err(CoreError::Type(format!(
            "cible inconnue « {t} » (cuda, opencl, wgsl ou ir)"
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
    Cu(Arc<CuDevice>),
    Cl(Arc<ClDevice>),
    Wg(Arc<WgDevice>),
    Cpu,
}

impl Device {
    /// Ouvre `cuda:<i>`, `opencl:<i>`, `metal:<i>`, `vulkan:<i>`, `dx12:<i>` ou `cpu:0`.
    pub fn open(id: &str) -> R<Device> {
        let (backend, index) = id.split_once(':').unwrap_or((id, "0"));
        let index: usize = index.parse().map_err(|_| {
            CoreError::Device(format!("identifiant de périphérique invalide : {id}"))
        })?;
        match backend {
            "cuda" => CuDevice::open(index).map(Device::Cu).map_err(dev_err),
            "opencl" => ClDevice::open(index).map(Device::Cl).map_err(dev_err),
            "metal" | "vulkan" | "dx12" => WgDevice::open(backend, index)
                .map(Device::Wg)
                .map_err(dev_err),
            "cpu" if index == 0 => Ok(Device::Cpu),
            _ => Err(CoreError::Device(format!("périphérique inconnu : {id}"))),
        }
    }

    /// Même périphérique ouvert (pas seulement le même identifiant : un périphérique perdu par le
    /// driver est rouvert sous le même identifiant).
    pub fn same(&self, other: &Device) -> bool {
        match (self, other) {
            (Device::Cu(a), Device::Cu(b)) => Arc::ptr_eq(a, b),
            (Device::Cl(a), Device::Cl(b)) => Arc::ptr_eq(a, b),
            (Device::Wg(a), Device::Wg(b)) => Arc::ptr_eq(a, b),
            (Device::Cpu, Device::Cpu) => true,
            _ => false,
        }
    }

    pub fn info(&self) -> DeviceInfo {
        match self {
            Device::Cu(d) => d.info.clone(),
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
            Device::Cu(_) => false,
            Device::Cl(d) => d.info.unified_memory,
            Device::Wg(d) => d.info.backend == "metal", // newBufferWithBytesNoCopy
            Device::Cpu => true,
        }
    }

    /// Attend la fin de tous les calculs en file.
    pub fn finish(&self) -> R<()> {
        match self {
            Device::Cu(d) => d.finish().map_err(dev_err),
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
    /// Variantes compilées à la demande : indices 64 bits (tableau de plus de 2³¹ − 1 éléments) et
    /// tailles connues à l'appel remplacées par des constantes (§5.4).
    pub(crate) variants: Arc<std::sync::Mutex<HashMap<VariantKey, Imp>>>,
}

/// (indices 64 bits, paramètres spécialisés).
pub(crate) type VariantKey = (bool, Vec<(usize, i64)>);

/// Variantes spécialisées gardées au plus par kernel et par périphérique ; au-delà (tailles qui
/// changent à chaque appel), la version générique sert, sans recompiler.
const MAX_SPECIALIZED: usize = 8;

impl Compiled {
    /// Implémentation à utiliser pour cet appel.
    pub(crate) fn variant(&self, dev: &Device, wide: bool, spec: Vec<(usize, i64)>) -> R<Imp> {
        if !wide && spec.is_empty() {
            return Ok(self.imp.clone());
        }
        let mut cache = self.variants.lock().unwrap_or_else(|e| e.into_inner());
        let key = (wide, spec);
        if let Some(imp) = cache.get(&key) {
            return Ok(imp.clone());
        }
        let key = if !key.1.is_empty()
            && cache
                .keys()
                .filter(|(w, s)| *w == wide && !s.is_empty())
                .count()
                >= MAX_SPECIALIZED
        {
            (wide, Vec::new()) // trop de tailles différentes : version générique
        } else {
            key
        };
        if !key.0 && key.1.is_empty() {
            return Ok(self.imp.clone());
        }
        if let Some(imp) = cache.get(&key) {
            return Ok(imp.clone());
        }
        let imp = build(dev, &self.kernel, key.0, &key.1)?;
        cache.insert(key, imp.clone());
        Ok(imp)
    }
}

#[derive(Clone)]
pub(crate) enum Imp {
    Cu(Arc<CuKernel>),
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

/// Débogage : avec `GPUSYNC_DUMP_DIR`, chaque code compilé est écrit dans ce dossier
/// (`<kernel>.<variante>.<cl|cu|wgsl>`).
fn dump(k: &Kernel, wide: bool, spec: &[(usize, i64)], ext: &str, code: &str) {
    if let Some(dir) = std::env::var_os("GPUSYNC_DUMP_DIR") {
        let tag: Vec<String> = spec
            .iter()
            .map(|(p, v)| format!("{}={v}", k.params[*p].name))
            .collect();
        let name = format!(
            "{}.{}{}.{ext}",
            k.name,
            if wide { "i64." } else { "" },
            if tag.is_empty() {
                "generique".into()
            } else {
                tag.join(",")
            }
        );
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(std::path::Path::new(&dir).join(name), code);
    }
}

/// Génère et compile une variante de `k` pour `dev`.
fn build(dev: &Device, k: &Kernel, wide: bool, spec: &[(usize, i64)]) -> R<Imp> {
    let gen_err = |e: gpusync_codegen::CodegenError| CoreError::Unsupported(e.0);
    Ok(match dev {
        Device::Cu(d) => {
            let src = gpusync_codegen::emit_c_variant(k, true, wide, spec).map_err(gen_err)?;
            dump(k, wide, spec, "cu", &src.code);
            Imp::Cu(d.compile(&src.code, &src.entry).map_err(dev_err)?)
        }
        Device::Cl(d) => {
            let src = gpusync_codegen::emit_c_variant(k, false, wide, spec).map_err(gen_err)?;
            dump(k, wide, spec, "cl", &src.code);
            Imp::Cl(d.compile(&src.code, &src.entry).map_err(dev_err)?)
        }
        // Un buffer wgpu ne dépasse pas quelques Gio : un tableau de plus de 2³¹ éléments ne peut pas
        // y être lié entier ; et WGSL n'a pas d'entiers 64 bits.
        Device::Wg(d) if wide => {
            return Err(CoreError::Unsupported(format!(
                "'{}' : un tableau de plus de 2³¹ − 1 éléments demande des indices 64 bits, \
                 absents de WGSL ({}) ; utilisez CUDA, OpenCL ou cpu:0",
                k.name, d.info.name
            )));
        }
        Device::Wg(d) => {
            let src = gpusync_codegen::emit_wgsl_variant(k, spec).map_err(gen_err)?;
            dump(k, wide, spec, "wgsl", &src.code);
            Imp::Wg(
                d.compile(
                    &src.code,
                    k.array_params().count(),
                    k.outputs.len(),
                    gpusync_codegen::has_f64_params(k),
                    &k.array_params()
                        .map(|(i, _)| k.writable().contains(&i))
                        .collect::<Vec<_>>(),
                )
                .map_err(dev_err)?,
            )
        }
        Device::Cpu => Imp::Cpu,
    })
}

/// Vérifie les capacités du périphérique (§6.2), génère le code et le compile.
pub fn compile(dev: &Device, k: &Arc<Kernel>) -> R<Compiled> {
    let info = dev.info();
    if let Some(msg) = unsupported(&info, k) {
        return Err(CoreError::Unsupported(msg));
    }
    Ok(Compiled {
        kernel: k.clone(),
        imp: build(dev, k, false, &[])?,
        variants: Default::default(),
    })
}

mod exec;
mod gpu;

pub use exec::{
    DeviceArray, Input, Mode, Storage, TransferLimits, launch, set_transfer_limits, transfer_limits,
};
