//! Moteur de transferts (ARCHITECTURE.md §9) : un seul chemin d'exécution, planifié en blocs.
//!
//! | Mode | Quand | Transferts |
//! |---|---|---|
//! | Direct | OpenCL sur mémoire unifiée | aucun : le GPU lit et écrit la mémoire de l'hôte |
//! | Résident | GPU dédié (ou wgpu), l'appel tient dans le budget | entrées envoyées par blocs ; résultat gardé sur le périphérique |
//! | Flux | au-delà du budget ou d'une taille de buffer | bloc par bloc, transferts recouverts par le calcul ; résultat écrit directement dans sa destination |
//!
//! Dans tous les modes, la mémoire hôte en plus est bornée (anneau de 3 blocs) : aucune copie
//! complète des données n'est faite en RAM.

use std::collections::VecDeque;
use std::sync::RwLock;

use gpusync_backend_opencl::{ClArg, ClBuffer, ClDevice, ClEvent, ClKernel, Pinned, wait};
use gpusync_backend_wgpu::{FluxIn, WgBuffer};
use gpusync_ir::interp::{self, Arg};
use gpusync_ir::{Const, Kernel, ParamKind, Scalar};
use gpusync_runtime::Backend;

use crate::{Compiled, CoreError, Device, Imp, R, dev_err};

/// Réglages du moteur de transferts (§9.2, §9.3), modifiables (tests, `gs.set_memory_budget`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransferLimits {
    /// Taille cible d'un bloc transféré, par tableau (octets).
    pub chunk_bytes: usize,
    /// Budget mémoire du périphérique (octets). `None` : 50 % de sa mémoire.
    pub budget_bytes: Option<u64>,
    /// Taille max d'un bloc en mode Direct (octets). `None` : taille max d'une allocation.
    pub direct_chunk_bytes: Option<usize>,
}

static LIMITS: RwLock<TransferLimits> = RwLock::new(TransferLimits {
    chunk_bytes: 64 << 20,
    budget_bytes: None,
    direct_chunk_bytes: None,
});

pub fn transfer_limits() -> TransferLimits {
    *LIMITS.read().unwrap_or_else(|e| e.into_inner())
}

pub fn set_transfer_limits(l: TransferLimits) {
    *LIMITS.write().unwrap_or_else(|e| e.into_inner()) = l;
}

/// Mode d'exécution retenu pour un appel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Direct,
    Resident,
    Flux,
    Cpu,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Direct => "direct",
            Mode::Resident => "resident",
            Mode::Flux => "flux",
            Mode::Cpu => "cpu",
        }
    }
}

/// Où se trouve le résultat d'un kernel.
pub enum Storage {
    /// Mode Direct (OpenCL, mémoire unifiée) : blocs adossés à la mémoire hôte `ptr`.
    ClDirect { bufs: Vec<ClBuffer>, ptr: *mut u8 },
    /// Mode Résident (OpenCL) : en mémoire du périphérique ; `done` signale la fin du calcul.
    Cl { buf: ClBuffer, done: ClEvent },
    /// Mode Résident (wgpu).
    Wg(WgBuffer),
    /// Déjà en mémoire hôte (mode Flux, interpréteur).
    Host { ptr: *mut u8 },
}

unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

/// Résultat d'un kernel.
pub struct DeviceArray {
    pub device: Device,
    pub storage: Storage,
    pub dtype: Scalar,
    pub len: usize,
    pub mode: Mode,
    /// Des buffers utilisent encore la mémoire de l'hôte : attendre la fin des calculs avant
    /// que l'appelant ne la libère.
    host_backed_io: bool,
}

impl DeviceArray {
    pub fn bytes(&self) -> usize {
        self.len * self.dtype.size()
    }

    /// Pointeur hôte si le résultat est (ou sera) directement dans la mémoire de l'hôte.
    pub fn host_ptr(&self) -> Option<*mut u8> {
        match &self.storage {
            Storage::ClDirect { ptr, .. } | Storage::Host { ptr } => Some(*ptr),
            _ => None,
        }
    }

    /// Attend la fin du calcul et place le résultat à `dst` (`bytes()` octets). Si `dst` est déjà la
    /// mémoire du résultat (Direct, Flux), seule la synchronisation a lieu.
    ///
    /// # Safety
    /// `dst` doit pointer vers `bytes()` octets accessibles en écriture.
    pub unsafe fn read_into(&self, dst: *mut u8) -> R<()> {
        let n = self.bytes();
        let chunk = transfer_limits().chunk_bytes;
        match (&self.storage, &self.device) {
            (Storage::ClDirect { bufs, ptr }, Device::Cl(d)) => {
                for b in bufs {
                    d.sync_host(b).map_err(dev_err)?;
                }
                if dst != *ptr {
                    unsafe { std::ptr::copy_nonoverlapping(*ptr, dst, n) };
                }
            }
            (Storage::Cl { buf, done }, Device::Cl(d)) => unsafe {
                cl_download(d, buf, done, dst, n, chunk)?
            },
            (Storage::Wg(buf), Device::Wg(d)) => unsafe {
                d.download(buf, 0, dst, n, chunk).map_err(dev_err)?
            },
            (Storage::Host { ptr }, _) => {
                if dst != *ptr {
                    unsafe { std::ptr::copy_nonoverlapping(*ptr, dst, n) };
                }
            }
            _ => {
                return Err(CoreError::Device(
                    "résultat incohérent avec son périphérique".into(),
                ));
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
    /// Mémoire de l'hôte (tableau NumPy contigu, memmap…), gardée en vie par l'appelant.
    Host { ptr: *const u8, bytes: usize },
    /// Résultat d'un kernel précédent.
    Device(&'a DeviceArray),
}

/// Entrée résolue pour le périphérique cible.
enum In<'a> {
    Host(*const u8),
    Dev(&'a DeviceArray),
}

/// Lance `c` sur `n` éléments ; `out_host` est la destination finale du résultat (`n` éléments).
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
    out_host: *mut u8,
) -> R<DeviceArray> {
    let k = &c.kernel;
    let arrays: Vec<_> = k.array_params().map(|(_, p)| p.ty).collect();
    let n_scalars = k.params.len() - arrays.len();
    if inputs.len() != arrays.len() || scalars.len() != n_scalars {
        return Err(CoreError::Type(format!(
            "'{}' attend {} tableau(x) et {} scalaire(s)",
            k.name,
            arrays.len(),
            n_scalars
        )));
    }
    for (input, (_, p)) in inputs.iter().zip(k.array_params()) {
        let bytes = match input {
            Input::Host { bytes, .. } => *bytes,
            Input::Device(a) => a.bytes(),
        };
        if bytes != n * p.ty.size() {
            return Err(CoreError::Shape(format!(
                "tableau '{}' : {bytes} octets pour {n} éléments de type {}",
                p.name, p.ty
            )));
        }
    }

    // Résolution des entrées : un résultat déjà en mémoire hôte est relu sur place ; un résultat
    // resté sur un autre périphérique repasse par l'hôte.
    let mut staged: Vec<Vec<u8>> = Vec::new();
    let mut idx_staged: Vec<Option<usize>> = Vec::new();
    for i in inputs {
        if let Input::Device(a) = i {
            if a.host_ptr().is_none() && a.device.id() != dev.id() {
                let mut v = vec![0u8; a.bytes()];
                unsafe { a.read_into(v.as_mut_ptr())? };
                staged.push(v);
                idx_staged.push(Some(staged.len() - 1));
                continue;
            }
        }
        idx_staged.push(None);
    }
    let mut ins: Vec<In> = Vec::with_capacity(inputs.len());
    for (i, st) in inputs.iter().zip(&idx_staged) {
        ins.push(match (i, st) {
            (_, Some(s)) => In::Host(staged[*s].as_ptr()),
            (Input::Host { ptr, .. }, None) => In::Host(*ptr),
            (Input::Device(a), None) => match a.host_ptr() {
                Some(p) => {
                    unsafe { a.read_into(p)? }; // synchronisation seulement
                    In::Host(p)
                }
                None => In::Dev(a),
            },
        });
    }

    let limits = transfer_limits();
    let out_elem = k.output.size();
    let result = match (dev, &c.imp) {
        (Device::Cpu, Imp::Cpu) => unsafe { run_cpu(k, &ins, &arrays, scalars, n, out_host) },
        (Device::Cl(d), Imp::Cl(ck)) if d.info.unified_memory => unsafe {
            cl_direct(d, ck, k, &ins, &arrays, scalars, n, out_host, &limits)
        },
        (Device::Cl(d), Imp::Cl(ck)) => {
            if fits(dev, &ins, &arrays, n, out_elem, &limits) {
                unsafe { cl_resident(d, ck, k, &ins, &arrays, scalars, n, &limits) }
            } else {
                unsafe { cl_flux(d, ck, k, &ins, &arrays, scalars, n, out_host, &limits) }
            }
        }
        (Device::Wg(d), Imp::Wg(wk)) => {
            let (words, fwords) = wg_params(k, scalars)?;
            if fits(dev, &ins, &arrays, n, out_elem, &limits) {
                let mut owned = Vec::new();
                for (i, ty) in ins.iter().zip(&arrays) {
                    if let In::Host(p) = i {
                        let b = d.alloc(n * ty.size()).map_err(dev_err)?;
                        unsafe {
                            d.upload(&b, 0, *p, n * ty.size(), limits.chunk_bytes)
                                .map_err(dev_err)?
                        };
                        owned.push(b);
                    }
                }
                let mut own = owned.iter();
                let bufs: Vec<&WgBuffer> = ins
                    .iter()
                    .map(|i| match i {
                        In::Host(_) => Ok(own.next().unwrap()),
                        In::Dev(a) => match &a.storage {
                            Storage::Wg(b) => Ok(b),
                            _ => Err(CoreError::Device(
                                "entrée sur un autre type de périphérique".into(),
                            )),
                        },
                    })
                    .collect::<R<_>>()?;
                let out = d.alloc(n * out_elem).map_err(dev_err)?;
                let mut w = words.clone();
                w.push(n as u32);
                d.launch(wk, &bufs, &out, &w, &fwords, n).map_err(dev_err)?;
                Ok((Storage::Wg(out), Mode::Resident, false))
            } else {
                let flux_in: Vec<FluxIn> = ins
                    .iter()
                    .zip(&arrays)
                    .map(|(i, ty)| match i {
                        In::Host(p) => Ok(FluxIn::Host {
                            ptr: *p,
                            elem: ty.size(),
                        }),
                        In::Dev(a) => match &a.storage {
                            Storage::Wg(b) => Ok(FluxIn::Dev {
                                buf: b,
                                elem: ty.size(),
                            }),
                            _ => Err(CoreError::Device(
                                "entrée sur un autre type de périphérique".into(),
                            )),
                        },
                    })
                    .collect::<R<_>>()?;
                let elem_max = arrays
                    .iter()
                    .map(|t| t.size())
                    .chain([out_elem])
                    .max()
                    .unwrap();
                let max_buf = d.info.max_buffer_bytes.unwrap_or(u64::MAX) as usize;
                let chunk_elems = (limits.chunk_bytes.min(max_buf) / elem_max).max(1);
                unsafe {
                    d.flux(
                        wk,
                        &flux_in,
                        out_host,
                        out_elem,
                        &words,
                        &fwords,
                        n,
                        chunk_elems,
                    )
                    .map_err(dev_err)?
                };
                Ok((Storage::Host { ptr: out_host }, Mode::Flux, false))
            }
        }
        _ => Err(CoreError::Device(
            "kernel compilé pour un autre périphérique".into(),
        )),
    };
    let (storage, mode, host_backed_io) = result?;
    // Entrées rapatriées d'un autre périphérique : copies locales libérées au retour.
    if !staged.is_empty() && host_backed_io {
        dev.finish()?;
    }
    Ok(DeviceArray {
        device: dev.clone(),
        storage,
        dtype: k.output,
        len: n,
        mode,
        host_backed_io: host_backed_io && staged.is_empty(),
    })
}

type Planned = R<(Storage, Mode, bool)>;

/// Mémoire du périphérique pour le budget : connue en OpenCL ; pour wgpu, celle du même GPU vu
/// par OpenCL s'il existe, sinon 4 Gio (prudent).
fn device_memory(dev: &Device) -> u64 {
    let info = dev.info();
    if let Some(m) = info.memory_bytes {
        return m;
    }
    gpusync_backend_opencl::OpenClBackend
        .devices()
        .ok()
        .and_then(|ds| {
            ds.into_iter()
                .find(|d| d.name == info.name)
                .and_then(|d| d.memory_bytes)
        })
        .unwrap_or(4 << 30)
}

/// L'appel tient-il en mode Résident (budget, taille max d'un buffer, une seule grille) ?
fn fits(
    dev: &Device,
    ins: &[In],
    arrays: &[Scalar],
    n: usize,
    out_elem: usize,
    limits: &TransferLimits,
) -> bool {
    let budget = limits.budget_bytes.unwrap_or(device_memory(dev) / 2);
    let max_buf = dev.info().max_buffer_bytes.unwrap_or(u64::MAX);
    let sizes: Vec<u64> = ins
        .iter()
        .zip(arrays)
        .map(|(_, t)| (n * t.size()) as u64)
        .chain([(n * out_elem) as u64])
        .collect();
    let new_bytes: u64 = ins
        .iter()
        .zip(arrays)
        .filter(|(i, _)| matches!(i, In::Host(_)))
        .map(|(_, t)| (n * t.size()) as u64)
        .sum::<u64>()
        + (n * out_elem) as u64;
    n <= u32::MAX as usize && new_bytes <= budget && sizes.iter().all(|s| *s <= max_buf)
}

fn scalar_args(k: &Kernel, scalars: &[Const]) -> Vec<Const> {
    let mut s = scalars.iter();
    k.params
        .iter()
        .filter(|p| p.kind == ParamKind::Scalar)
        .map(|p| interp::convert(*s.next().unwrap(), p.ty))
        .collect()
}

fn wg_params(k: &Kernel, scalars: &[Const]) -> R<(Vec<u32>, Vec<f64>)> {
    let (mut w, mut f) = (Vec::new(), Vec::new());
    for c in scalar_args(k, scalars) {
        match c {
            Const::I32(v) => w.push(v as u32),
            Const::U32(v) => w.push(v),
            Const::F32(v) => w.push(v.to_bits()),
            Const::Bool(v) => w.push(v as u32),
            Const::F64(v) => f.push(v),
        }
    }
    Ok((w, f))
}

fn cl_scalar(c: Const) -> ClArg<'static> {
    match c {
        Const::I32(v) => ClArg::I32(v),
        Const::U32(v) => ClArg::U32(v),
        Const::F32(v) => ClArg::F32(v),
        Const::F64(v) => ClArg::F64(v),
        Const::Bool(v) => ClArg::I32(v as i32),
    }
}

/// Arguments OpenCL dans l'ordre des paramètres : tableaux = `bufs` (dans l'ordre), puis sortie et n.
fn cl_args<'a>(
    k: &Kernel,
    bufs: &[&'a ClBuffer],
    scalars: &[Const],
    out: &'a ClBuffer,
    n: u32,
) -> Vec<ClArg<'a>> {
    let mut b = bufs.iter();
    let mut s = scalar_args(k, scalars).into_iter();
    let mut args: Vec<ClArg> = k
        .params
        .iter()
        .map(|p| match p.kind {
            ParamKind::Array => ClArg::Buf(b.next().unwrap()),
            ParamKind::Scalar => cl_scalar(s.next().unwrap()),
        })
        .collect();
    args.push(ClArg::Buf(out));
    args.push(ClArg::U32(n));
    args
}

unsafe fn run_cpu(
    k: &Kernel,
    ins: &[In],
    arrays: &[Scalar],
    scalars: &[Const],
    n: usize,
    out: *mut u8,
) -> Planned {
    let slices: Vec<&[u8]> = ins
        .iter()
        .zip(arrays)
        .map(|(i, t)| match i {
            In::Host(p) => Ok(unsafe { std::slice::from_raw_parts(*p, n * t.size()) }),
            In::Dev(_) => Err(CoreError::Device(
                "cpu:0 : entrée restée sur un périphérique".into(),
            )),
        })
        .collect::<R<_>>()?;
    let mut sl = slices.into_iter();
    let mut sc = scalar_args(k, scalars).into_iter();
    let args: Vec<Arg> = k
        .params
        .iter()
        .map(|p| match p.kind {
            ParamKind::Array => Arg::Array(sl.next().unwrap()),
            ParamKind::Scalar => Arg::Scalar(sc.next().unwrap()),
        })
        .collect();
    interp::run(k, n, &args, unsafe {
        std::slice::from_raw_parts_mut(out, n * k.output.size())
    })
    .map_err(CoreError::Device)?;
    Ok((Storage::Host { ptr: out }, Mode::Cpu, false))
}

/// Mode Direct (OpenCL, mémoire unifiée) : chaque bloc utilise directement la mémoire de l'hôte.
#[allow(clippy::too_many_arguments)]
unsafe fn cl_direct(
    d: &ClDevice,
    ck: &ClKernel,
    k: &Kernel,
    ins: &[In],
    arrays: &[Scalar],
    scalars: &[Const],
    n: usize,
    out: *mut u8,
    limits: &TransferLimits,
) -> Planned {
    let elem_max = arrays
        .iter()
        .map(|t| t.size())
        .chain([k.output.size()])
        .max()
        .unwrap();
    let max_alloc = d
        .info
        .max_buffer_bytes
        .unwrap_or(u64::MAX)
        .min(usize::MAX as u64) as usize;
    let chunk_bytes = limits
        .direct_chunk_bytes
        .unwrap_or(max_alloc)
        .min(max_alloc);
    let chunk = (chunk_bytes / elem_max).clamp(1, u32::MAX as usize);
    let mut outs = Vec::new();
    let mut start = 0;
    while start < n {
        let len = chunk.min(n - start);
        let mut bufs = Vec::new();
        for (i, ty) in ins.iter().zip(arrays) {
            let In::Host(p) = i else {
                return Err(CoreError::Device(
                    "mode Direct : entrée restée sur un périphérique".into(),
                ));
            };
            bufs.push(
                unsafe { d.wrap_host(p.add(start * ty.size()) as *mut u8, len * ty.size()) }
                    .map_err(dev_err)?,
            );
        }
        let ob = unsafe { d.wrap_host(out.add(start * k.output.size()), len * k.output.size()) }
            .map_err(dev_err)?;
        let refs: Vec<&ClBuffer> = bufs.iter().collect();
        d.launch(ck, &cl_args(k, &refs, scalars, &ob, len as u32), len, &[])
            .map_err(dev_err)?;
        outs.push(ob);
        start += len;
    }
    Ok((
        Storage::ClDirect {
            bufs: outs,
            ptr: out,
        },
        Mode::Direct,
        true,
    ))
}

/// Anneau de 3 tampons épinglés pour envoyer `bytes` octets de `src` vers `dst` (file de transfert).
/// Renvoie l'événement de la dernière écriture.
unsafe fn cl_upload(
    d: &ClDevice,
    dst: &ClBuffer,
    src: *const u8,
    bytes: usize,
    chunk: usize,
) -> R<Option<ClEvent>> {
    let chunk = chunk.min(bytes).max(4);
    let mut ring: Vec<(Pinned, Option<ClEvent>)> = Vec::new();
    for _ in 0..3.min(bytes.div_ceil(chunk)) {
        ring.push((d.pinned(chunk).map_err(dev_err)?, None));
    }
    let mut last = None;
    let mut off = 0;
    let mut i = 0;
    while off < bytes {
        let len = chunk.min(bytes - off);
        let slot = i % ring.len();
        if let Some(e) = ring[slot].1.take() {
            wait(&e).map_err(dev_err)?; // le tampon est réutilisable
        }
        let p = &ring[slot].0;
        unsafe { std::ptr::copy_nonoverlapping(src.add(off), p.ptr, len) };
        let e = unsafe { d.write_at(dst, off, p.ptr, len, &[]) }.map_err(dev_err)?;
        d.flush().map_err(dev_err)?;
        ring[slot].1 = Some(e);
        off += len;
        i += 1;
    }
    for (p, e) in ring {
        if let Some(e) = e {
            wait(&e).map_err(dev_err)?;
            last = Some(e);
        }
        d.recycle(p);
    }
    Ok(last)
}

/// Rapatrie un résultat Résident par l'anneau épinglé, après `done`.
unsafe fn cl_download(
    d: &ClDevice,
    src: &ClBuffer,
    done: &ClEvent,
    dst: *mut u8,
    bytes: usize,
    chunk: usize,
) -> R<()> {
    let chunk = chunk.min(bytes).max(4);
    let mut ring: Vec<Pinned> = Vec::new();
    for _ in 0..3.min(bytes.div_ceil(chunk)) {
        ring.push(d.pinned(chunk).map_err(dev_err)?);
    }
    let mut pending: VecDeque<(usize, usize, usize, ClEvent)> = VecDeque::new();
    let mut off = 0;
    let mut i = 0;
    loop {
        if pending.len() == ring.len() || (off >= bytes && !pending.is_empty()) {
            let (slot, o, len, e) = pending.pop_front().unwrap();
            wait(&e).map_err(dev_err)?;
            unsafe { std::ptr::copy_nonoverlapping(ring[slot].ptr, dst.add(o), len) };
            continue;
        }
        if off >= bytes {
            break;
        }
        let len = chunk.min(bytes - off);
        let slot = i % ring.len();
        let e = unsafe { d.read_at(src, off, ring[slot].ptr, len, &[done]) }.map_err(dev_err)?;
        d.flush().map_err(dev_err)?;
        pending.push_back((slot, off, len, e));
        off += len;
        i += 1;
    }
    for p in ring {
        d.recycle(p);
    }
    Ok(())
}

/// Mode Résident (OpenCL, GPU dédié) : entrées envoyées par l'anneau épinglé, résultat gardé en VRAM.
#[allow(clippy::too_many_arguments)]
unsafe fn cl_resident(
    d: &ClDevice,
    ck: &ClKernel,
    k: &Kernel,
    ins: &[In],
    arrays: &[Scalar],
    scalars: &[Const],
    n: usize,
    limits: &TransferLimits,
) -> Planned {
    let mut owned = Vec::new();
    let mut last: Option<ClEvent> = None;
    for (i, ty) in ins.iter().zip(arrays) {
        if let In::Host(p) = i {
            let b = d.alloc(n * ty.size()).map_err(dev_err)?;
            if let Some(e) = unsafe { cl_upload(d, &b, *p, n * ty.size(), limits.chunk_bytes)? } {
                last = Some(e);
            }
            owned.push(b);
        }
    }
    let mut own = owned.iter();
    let bufs: Vec<&ClBuffer> = ins
        .iter()
        .map(|i| match i {
            In::Host(_) => Ok(own.next().unwrap()),
            In::Dev(a) => match &a.storage {
                Storage::Cl { buf, .. } => Ok(buf),
                _ => Err(CoreError::Device(
                    "entrée sur un autre type de périphérique".into(),
                )),
            },
        })
        .collect::<R<_>>()?;
    let out = d.alloc(n * k.output.size()).map_err(dev_err)?;
    let waits: Vec<&ClEvent> = last.iter().collect();
    let done = d
        .launch(ck, &cl_args(k, &bufs, scalars, &out, n as u32), n, &waits)
        .map_err(dev_err)?;
    Ok((Storage::Cl { buf: out, done }, Mode::Resident, false))
}

/// Mode Flux (OpenCL, GPU dédié) : pipeline à 3 emplacements sur deux files ; pendant le calcul du
/// bloc k, le bloc k+1 est envoyé et le bloc k−1 revient dans sa destination finale.
#[allow(clippy::too_many_arguments)]
unsafe fn cl_flux(
    d: &ClDevice,
    ck: &ClKernel,
    k: &Kernel,
    ins: &[In],
    arrays: &[Scalar],
    scalars: &[Const],
    n: usize,
    out: *mut u8,
    limits: &TransferLimits,
) -> Planned {
    const SLOTS: usize = 3;
    let out_elem = k.output.size();
    let elem_max = arrays
        .iter()
        .map(|t| t.size())
        .chain([out_elem])
        .max()
        .unwrap();
    let max_alloc = d.info.max_buffer_bytes.unwrap_or(u64::MAX) as usize;
    let chunk = (limits.chunk_bytes.min(max_alloc) / elem_max).clamp(1, n);
    struct Slot {
        ins: Vec<ClBuffer>,
        pins: Vec<Option<Pinned>>,
        out: ClBuffer,
        pin_out: Pinned,
        busy: Option<(usize, usize, ClEvent)>,
    }
    let mut ring: Vec<Slot> = Vec::new();
    for _ in 0..SLOTS.min(n.div_ceil(chunk)) {
        let mut sins = Vec::new();
        let mut pins = Vec::new();
        for (i, ty) in ins.iter().zip(arrays) {
            sins.push(d.alloc(chunk * ty.size()).map_err(dev_err)?);
            pins.push(match i {
                In::Host(_) => Some(d.pinned(chunk * ty.size()).map_err(dev_err)?),
                In::Dev(_) => None,
            });
        }
        ring.push(Slot {
            ins: sins,
            pins,
            out: d.alloc(chunk * out_elem).map_err(dev_err)?,
            pin_out: d.pinned(chunk * out_elem).map_err(dev_err)?,
            busy: None,
        });
    }
    let finish = |s: &mut Slot| -> R<()> {
        if let Some((start, len, e)) = s.busy.take() {
            wait(&e).map_err(dev_err)?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    s.pin_out.ptr,
                    out.add(start * out_elem),
                    len * out_elem,
                )
            };
        }
        Ok(())
    };
    let mut start = 0;
    let mut i = 0;
    while start < n {
        let len = chunk.min(n - start);
        let slot = i % ring.len();
        finish(&mut ring[slot])?;
        let s = &mut ring[slot];
        let mut writes: Vec<ClEvent> = Vec::new();
        for (a, (input, ty)) in ins.iter().zip(arrays).enumerate() {
            let bytes = len * ty.size();
            match input {
                In::Host(p) => {
                    let pin = s.pins[a].as_ref().unwrap();
                    unsafe {
                        std::ptr::copy_nonoverlapping(p.add(start * ty.size()), pin.ptr, bytes)
                    };
                    writes.push(
                        unsafe { d.write_at(&s.ins[a], 0, pin.ptr, bytes, &[]) }
                            .map_err(dev_err)?,
                    );
                }
                In::Dev(arr) => match &arr.storage {
                    Storage::Cl { buf, done } => {
                        writes.push(
                            d.copy_at(buf, start * ty.size(), &s.ins[a], 0, bytes, &[done])
                                .map_err(dev_err)?,
                        );
                    }
                    _ => {
                        return Err(CoreError::Device(
                            "entrée sur un autre type de périphérique".into(),
                        ));
                    }
                },
            }
        }
        d.flush().map_err(dev_err)?;
        let refs: Vec<&ClBuffer> = s.ins.iter().collect();
        let wait_w: Vec<&ClEvent> = writes.iter().collect();
        let ek = d
            .launch(
                ck,
                &cl_args(k, &refs, scalars, &s.out, len as u32),
                len,
                &wait_w,
            )
            .map_err(dev_err)?;
        let er = unsafe { d.read_at(&s.out, 0, s.pin_out.ptr, len * out_elem, &[&ek]) }
            .map_err(dev_err)?;
        d.flush().map_err(dev_err)?;
        s.busy = Some((start, len, er));
        start += len;
        i += 1;
    }
    let used = ring.len();
    for j in 0..used {
        finish(&mut ring[(i + j) % used])?;
    }
    for s in ring {
        for p in s.pins.into_iter().flatten() {
            d.recycle(p);
        }
        d.recycle(s.pin_out);
    }
    Ok((Storage::Host { ptr: out }, Mode::Flux, false))
}
