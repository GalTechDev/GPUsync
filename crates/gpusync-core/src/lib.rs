//! Cœur Rust de GPUsync (sans Python) : frontend + génération de code + exécution sur un backend.
//!
//! - [`parse`] : AST Python (JSON) → IR typée ;
//! - [`Device`] : périphérique ouvert (`opencl:0`, `metal:0`, `vulkan:1`, `dx12:0`, `cpu:0`) ;
//! - [`compile`] : vérifie les capacités, génère et compile le code (avec cache) ;
//! - [`launch`] : exécute sur `n` éléments et renvoie un [`DeviceArray`] (résultat resté sur le
//!   périphérique jusqu'à [`DeviceArray::read_into`]).
//!
//! Mémoire unifiée (§8.4) : sur un périphérique OpenCL à mémoire unifiée, entrées et sortie
//! utilisent directement la mémoire de l'hôte (aucune copie).

use std::sync::Arc;

use gpusync_backend_opencl::{ClArg, ClBuffer, ClDevice, ClKernel};
use gpusync_backend_wgpu::{WgBuffer, WgDevice, WgKernel};
use gpusync_ir::interp::{self, Arg};
use gpusync_ir::{Const, Kernel, ParamKind, Scalar};
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
            Device::Wg(_) => false,
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
    imp: Imp,
}

#[derive(Clone)]
enum Imp {
    Cl(Arc<ClKernel>),
    Wg(Arc<WgKernel>),
    Cpu,
}

/// Vérifie les capacités du périphérique (§6.2), génère le code et le compile.
pub fn compile(dev: &Device, k: &Arc<Kernel>) -> R<Compiled> {
    let info = dev.info();
    if k.requirements().f64 && !info.f64 {
        let hint = match dev {
            Device::Wg(_) => {
                " (WGSL n'a pas de f64 ; essayez un périphérique OpenCL qui le supporte, ou cpu:0)"
            }
            _ => " (essayez un autre périphérique, ou cpu:0)",
        };
        return Err(CoreError::Unsupported(format!(
            "le kernel '{}' utilise f64, non supporté par '{}' ({}){hint}",
            k.name,
            info.name,
            info.id()
        )));
    }
    let imp = match dev {
        Device::Cl(d) => {
            let src = gpusync_codegen::emit_opencl(k).map_err(|e| CoreError::Unsupported(e.0))?;
            Imp::Cl(d.compile(&src.code, &src.entry).map_err(dev_err)?)
        }
        Device::Wg(d) => {
            let src = gpusync_codegen::emit_wgsl(k).map_err(|e| CoreError::Unsupported(e.0))?;
            Imp::Wg(
                d.compile(&src.code, k.array_params().count())
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

/// Données d'un tableau sur un périphérique.
pub enum Storage {
    Cl(ClBuffer),
    Wg(WgBuffer),
    /// Mémoire de l'hôte (interpréteur) : appartient à l'appelant, qui la garde en vie.
    Host {
        ptr: *mut u8,
        bytes: usize,
    },
}

unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

/// Résultat d'un kernel, resté sur le périphérique.
pub struct DeviceArray {
    pub device: Device,
    pub storage: Storage,
    pub dtype: Scalar,
    pub len: usize,
    /// Des buffers utilisent la mémoire de l'hôte : il faut attendre la fin des calculs avant
    /// que l'appelant ne libère cette mémoire.
    host_backed_io: bool,
}

impl DeviceArray {
    pub fn bytes(&self) -> usize {
        self.len * self.dtype.size()
    }

    /// Pointeur hôte si le résultat est déjà en mémoire de l'hôte (zéro-copie).
    pub fn host_ptr(&self) -> Option<*mut u8> {
        match &self.storage {
            Storage::Cl(b) if b.is_host_backed() => Some(b.host_ptr),
            Storage::Host { ptr, .. } => Some(*ptr),
            _ => None,
        }
    }

    /// Attend la fin du calcul et copie le résultat vers `dst` (`bytes()` octets).
    /// Si `dst` est la mémoire hôte du résultat (zéro-copie), seule la synchronisation a lieu.
    ///
    /// # Safety
    /// `dst` doit pointer vers `bytes()` octets accessibles en écriture.
    pub unsafe fn read_into(&self, dst: *mut u8) -> R<()> {
        let n = self.bytes();
        match &self.storage {
            Storage::Cl(b) if b.is_host_backed() => {
                let Device::Cl(d) = &self.device else {
                    unreachable!()
                };
                d.sync_host(b).map_err(dev_err)?;
                if dst != b.host_ptr {
                    unsafe { std::ptr::copy_nonoverlapping(b.host_ptr, dst, n) };
                }
            }
            Storage::Cl(b) => {
                let Device::Cl(d) = &self.device else {
                    unreachable!()
                };
                d.read(b, unsafe { std::slice::from_raw_parts_mut(dst, n) })
                    .map_err(dev_err)?;
            }
            Storage::Wg(b) => {
                let Device::Wg(d) = &self.device else {
                    unreachable!()
                };
                d.read(b, unsafe { std::slice::from_raw_parts_mut(dst, n) })
                    .map_err(dev_err)?;
            }
            Storage::Host { ptr, .. } => {
                if dst != *ptr {
                    unsafe { std::ptr::copy_nonoverlapping(*ptr, dst, n) };
                }
            }
        }
        Ok(())
    }
}

impl Drop for DeviceArray {
    fn drop(&mut self) {
        if self.host_backed_io {
            let _ = self.device.finish();
        }
    }
}

/// Un tableau d'entrée.
pub enum Input<'a> {
    /// Mémoire de l'hôte (tableau NumPy contigu), gardée en vie par l'appelant pendant le calcul.
    Host { ptr: *const u8, bytes: usize },
    /// Résultat d'un kernel précédent.
    Device(&'a DeviceArray),
}

/// Lance `c` sur `n` éléments.
///
/// - `inputs` : un par paramètre tableau, dans l'ordre ;
/// - `scalars` : un par paramètre scalaire, dans l'ordre (convertis au type déclaré) ;
/// - `out_host` : mémoire hôte de `n` éléments du type de sortie, utilisée directement si le
///   périphérique est en zéro-copie (obligatoire pour `cpu:0`).
///
/// # Safety
/// Les pointeurs hôte doivent rester valides jusqu'à la fin du calcul (lecture du résultat ou
/// destruction du `DeviceArray` renvoyé).
pub unsafe fn launch(
    dev: &Device,
    c: &Compiled,
    inputs: &[Input],
    scalars: &[Const],
    n: usize,
    out_host: Option<*mut u8>,
) -> R<DeviceArray> {
    let k = &c.kernel;
    let arrays: Vec<_> = k.array_params().collect();
    let n_scalars = k.params.len() - arrays.len();
    if inputs.len() != arrays.len() || scalars.len() != n_scalars {
        return Err(CoreError::Type(format!(
            "'{}' attend {} tableau(x) et {} scalaire(s)",
            k.name,
            arrays.len(),
            n_scalars
        )));
    }
    for (input, (_, p)) in inputs.iter().zip(&arrays) {
        let bytes = match input {
            Input::Host { bytes, .. } => *bytes,
            Input::Device(a) => a.bytes(),
        };
        if bytes != n * p.ty.size() {
            return Err(CoreError::Shape(format!(
                "tableau '{}' : {} octets pour {n} éléments de type {}",
                p.name, bytes, p.ty
            )));
        }
    }
    let n32 = u32::try_from(n).map_err(|_| {
        CoreError::Shape(format!(
            "{n} éléments : plus de 2³² éléments n'est pas supporté"
        ))
    })?;
    let out_bytes = n * k.output.size();

    // Une entrée calculée sur un autre périphérique repasse par l'hôte.
    let mut staged: Vec<Vec<u8>> = Vec::new();
    let mut resolved: Vec<Input> = Vec::with_capacity(inputs.len());
    for input in inputs {
        match input {
            Input::Device(a)
                if a.device.id() != dev.id()
                    || (matches!(dev, Device::Cpu) && a.host_ptr().is_none()) =>
            {
                let mut v = vec![0u8; a.bytes()];
                unsafe { a.read_into(v.as_mut_ptr())? };
                staged.push(v);
                let last = staged.last().unwrap();
                resolved.push(Input::Host {
                    ptr: last.as_ptr(),
                    bytes: last.len(),
                });
            }
            Input::Device(a) => resolved.push(Input::Device(a)),
            Input::Host { ptr, bytes } => resolved.push(Input::Host {
                ptr: *ptr,
                bytes: *bytes,
            }),
        }
    }

    match (dev, &c.imp) {
        (Device::Cpu, Imp::Cpu) => {
            let out = out_host
                .ok_or_else(|| CoreError::Device("cpu:0 : mémoire de sortie manquante".into()))?;
            let mut ins = resolved.iter().map(|i| match i {
                Input::Host { ptr, bytes } => unsafe { std::slice::from_raw_parts(*ptr, *bytes) },
                Input::Device(a) => unsafe {
                    std::slice::from_raw_parts(a.host_ptr().unwrap(), a.bytes())
                },
            });
            let mut s = scalars.iter();
            let args: Vec<Arg> = k
                .params
                .iter()
                .map(|p| match p.kind {
                    ParamKind::Array => Arg::Array(ins.next().unwrap()),
                    ParamKind::Scalar => Arg::Scalar(interp::convert(*s.next().unwrap(), p.ty)),
                })
                .collect();
            interp::run(k, n, &args, unsafe {
                std::slice::from_raw_parts_mut(out, out_bytes)
            })
            .map_err(CoreError::Device)?;
            Ok(DeviceArray {
                device: Device::Cpu,
                storage: Storage::Host {
                    ptr: out,
                    bytes: out_bytes,
                },
                dtype: k.output,
                len: n,
                host_backed_io: false,
            })
        }
        (Device::Cl(d), Imp::Cl(ck)) => {
            let zero_copy = d.info.unified_memory;
            let mut owned: Vec<ClBuffer> = Vec::new();
            let mut host_backed_io = false;
            for i in &resolved {
                if let Input::Host { ptr, bytes } = i {
                    let b = if zero_copy {
                        host_backed_io = true;
                        unsafe { d.wrap_host(*ptr as *mut u8, *bytes) }
                    } else {
                        // Copie non bloquante : l'entrée reste en vie jusqu'à la fin du calcul.
                        host_backed_io = true;
                        d.alloc(*bytes).and_then(|b| {
                            unsafe { d.write_async(&b, std::slice::from_raw_parts(*ptr, *bytes)) }
                                .map(|_| b)
                        })
                    }
                    .map_err(dev_err)?;
                    owned.push(b);
                }
            }
            let out = match out_host {
                Some(p) if zero_copy => {
                    host_backed_io = true;
                    unsafe { d.wrap_host(p, out_bytes) }
                }
                _ => d.alloc(out_bytes),
            }
            .map_err(dev_err)?;
            let mut own = owned.iter();
            let mut dev_in = resolved.iter();
            let mut s = scalars.iter();
            let mut args: Vec<ClArg> = Vec::with_capacity(k.params.len() + 2);
            for p in &k.params {
                match p.kind {
                    ParamKind::Array => match dev_in.next().unwrap() {
                        Input::Host { .. } => args.push(ClArg::Buf(own.next().unwrap())),
                        Input::Device(a) => match &a.storage {
                            Storage::Cl(b) => args.push(ClArg::Buf(b)),
                            _ => {
                                return Err(CoreError::Device(
                                    "entrée sur un autre type de périphérique".into(),
                                ));
                            }
                        },
                    },
                    ParamKind::Scalar => {
                        args.push(match interp::convert(*s.next().unwrap(), p.ty) {
                            Const::I32(v) => ClArg::I32(v),
                            Const::U32(v) => ClArg::U32(v),
                            Const::F32(v) => ClArg::F32(v),
                            Const::F64(v) => ClArg::F64(v),
                            Const::Bool(v) => ClArg::I32(v as i32),
                        })
                    }
                }
            }
            args.push(ClArg::Buf(&out));
            args.push(ClArg::U32(n32));
            d.launch(ck, &args, n).map_err(dev_err)?;
            drop(args);
            // Les buffers d'entrée peuvent être libérés : OpenCL les retient jusqu'à la fin du calcul.
            drop(owned);
            if !staged.is_empty() {
                // Entrées rapatriées d'un autre périphérique : copies locales libérées au retour.
                d.finish().map_err(dev_err)?;
            }
            Ok(DeviceArray {
                device: dev.clone(),
                storage: Storage::Cl(out),
                dtype: k.output,
                len: n,
                host_backed_io,
            })
        }
        (Device::Wg(d), Imp::Wg(wk)) => {
            // wgpu limite la taille d'un buffer lié à un kernel (1 Gio − 4 o sur certains GPU Intel).
            if let Some(max) = d.info.max_buffer_bytes {
                let biggest = resolved
                    .iter()
                    .map(|i| match i {
                        Input::Host { bytes, .. } => *bytes,
                        Input::Device(a) => a.bytes(),
                    })
                    .chain([out_bytes])
                    .max()
                    .unwrap_or(0);
                if biggest as u64 > max {
                    return Err(CoreError::Unsupported(format!(
                        "tableau de {:.2} Gio : dépasse la taille maximale d'un buffer sur {} ({}), {:.2} Gio. \
                         Utilisez un périphérique OpenCL, ou découpez les données (out-of-core, jalon M3)",
                        biggest as f64 / (1u64 << 30) as f64,
                        d.info.name,
                        d.info.id(),
                        max as f64 / (1u64 << 30) as f64
                    )));
                }
            }
            let mut owned: Vec<WgBuffer> = Vec::new();
            for i in &resolved {
                if let Input::Host { ptr, bytes } = i {
                    let b = d.alloc(*bytes).map_err(dev_err)?;
                    d.write(&b, unsafe { std::slice::from_raw_parts(*ptr, *bytes) })
                        .map_err(dev_err)?;
                    owned.push(b);
                }
            }
            let mut own = owned.iter();
            let mut bufs: Vec<&WgBuffer> = Vec::new();
            for i in &resolved {
                bufs.push(match i {
                    Input::Host { .. } => own.next().unwrap(),
                    Input::Device(a) => match &a.storage {
                        Storage::Wg(b) => b,
                        _ => {
                            return Err(CoreError::Device(
                                "entrée sur un autre type de périphérique".into(),
                            ));
                        }
                    },
                });
            }
            let mut params: Vec<u32> = Vec::new();
            let mut s = scalars.iter();
            for p in k.params.iter().filter(|p| p.kind == ParamKind::Scalar) {
                params.push(match interp::convert(*s.next().unwrap(), p.ty) {
                    Const::I32(v) => v as u32,
                    Const::U32(v) => v,
                    Const::F32(v) => v.to_bits(),
                    Const::Bool(v) => v as u32,
                    Const::F64(_) => {
                        return Err(CoreError::Unsupported("f64 indisponible en WGSL".into()));
                    }
                });
            }
            params.push(n32);
            let out = d.alloc(out_bytes).map_err(dev_err)?;
            d.launch(wk, &bufs, &out, &params, n).map_err(dev_err)?;
            Ok(DeviceArray {
                device: dev.clone(),
                storage: Storage::Wg(out),
                dtype: k.output,
                len: n,
                host_backed_io: false,
            })
        }
        _ => Err(CoreError::Device(
            "kernel compilé pour un autre périphérique".into(),
        )),
    }
}
