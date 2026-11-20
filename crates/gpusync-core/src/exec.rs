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
use std::sync::{Arc, RwLock};

use gpusync_backend_opencl::{ClArg, ClBuffer, ClDevice, ClEvent, ClKernel, Pinned, wait};
use gpusync_backend_wgpu::{FluxIn, WgBuffer};
use gpusync_ir::interp::{self, Arg};
use gpusync_ir::{Const, Grid, Kernel, ParamKind, Scalar};
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
    Cl { buf: ClBuffer, done: Arc<ClEvent> },
    /// Mode Résident (wgpu).
    Wg(WgBuffer),
    /// Mode Direct (Metal, `newBufferWithBytesNoCopy`) : blocs adossés à la mémoire hôte `ptr`.
    WgDirect { bufs: Vec<WgBuffer>, ptr: *mut u8 },
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
            Storage::ClDirect { ptr, .. }
            | Storage::WgDirect { ptr, .. }
            | Storage::Host { ptr } => Some(*ptr),
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
            (Storage::WgDirect { ptr, .. }, Device::Wg(d)) => {
                d.finish().map_err(dev_err)?;
                if dst != *ptr {
                    unsafe { std::ptr::copy_nonoverlapping(*ptr, dst, n) };
                }
            }
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
    Host(*const u8, usize),
    Dev(&'a DeviceArray),
}

impl In<'_> {
    fn bytes(&self) -> usize {
        match self {
            In::Host(_, b) => *b,
            In::Dev(a) => a.bytes(),
        }
    }
}

/// Lance `c` sur une grille de `n` éléments (taille de chaque sortie) ; `outs` sont les
/// destinations finales des sorties (`n` éléments chacune), dans l'ordre du kernel.
///
/// Kernel élément par élément : chaque entrée a `n` éléments. Kernel avec index : les entrées ont
/// leur propre taille (les dimensions sont passées en paramètres cachés, dans `scalars`).
///
/// # Safety
/// Les pointeurs hôte doivent rester valides jusqu'à la fin du calcul (lecture des résultats ou
/// destruction des `DeviceArray` renvoyés).
pub unsafe fn launch(
    dev: &Device,
    c: &Compiled,
    inputs: &[Input],
    scalars: &[Const],
    n: usize,
    outs: &[*mut u8],
) -> R<Vec<DeviceArray>> {
    let k = &c.kernel;
    let arrays: Vec<_> = k.array_params().map(|(_, p)| p.ty).collect();
    let n_engine = k.engine_params().iter().flatten().count();
    let n_scalars = k.params.len() - arrays.len() - n_engine;
    if inputs.len() != arrays.len() || scalars.len() != n_scalars {
        return Err(CoreError::Type(format!(
            "'{}' attend {} tableau(x) et {} scalaire(s)",
            k.name,
            arrays.len(),
            n_scalars
        )));
    }
    if outs.len() != k.outputs.len() {
        return Err(CoreError::Type(format!(
            "'{}' a {} sortie(s), {} destination(s) fournie(s)",
            k.name,
            k.outputs.len(),
            outs.len()
        )));
    }
    let indexed = matches!(k.grid, Grid::Indexed { .. });
    for (input, (_, p)) in inputs.iter().zip(k.array_params()) {
        let bytes = match input {
            Input::Host { bytes, .. } => *bytes,
            Input::Device(a) => a.bytes(),
        };
        let ok = if indexed {
            bytes % p.ty.size() == 0
        } else {
            bytes == n * p.ty.size()
        };
        if !ok {
            return Err(CoreError::Shape(format!(
                "tableau '{}' : {bytes} octets pour {} éléments de type {}",
                p.name,
                if indexed {
                    "des".to_string()
                } else {
                    n.to_string()
                },
                p.ty
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
            (_, Some(s)) => In::Host(staged[*s].as_ptr(), staged[*s].len()),
            (Input::Host { ptr, bytes }, None) => In::Host(*ptr, *bytes),
            (Input::Device(a), None) => match a.host_ptr() {
                Some(p) => {
                    unsafe { a.read_into(p)? }; // synchronisation seulement
                    In::Host(p, a.bytes())
                }
                None => In::Dev(a),
            },
        });
    }

    let limits = transfer_limits();
    let out_elems: Vec<usize> = k.outputs.iter().map(|t| t.size()).collect();
    let scalars = &full_scalars(k, scalars);
    let result: Planned = if n == 0 {
        Ok((
            outs.iter().map(|p| Storage::Host { ptr: *p }).collect(),
            Mode::Cpu,
            false,
        ))
    } else if indexed {
        let rows = Rows::new(k, scalars, n, &ins, &out_elems);
        match (dev, &c.imp) {
            (Device::Cpu, Imp::Cpu) => unsafe { run_cpu(k, &ins, scalars, n, outs) },
            (Device::Cl(d), Imp::Cl(ck)) if d.info.unified_memory => unsafe {
                rows_cl_direct(d, ck, k, &rows, &ins, scalars, outs, &limits)
            },
            (Device::Cl(d), Imp::Cl(ck)) => unsafe {
                rows_cl(
                    d,
                    ck,
                    k,
                    &rows,
                    &ins,
                    scalars,
                    outs,
                    &limits,
                    fits(dev, &ins, n, &out_elems, &limits),
                    budget(dev, &limits),
                )
            },
            (Device::Wg(d), Imp::Wg(wk)) => unsafe {
                match rows_wg_direct(d, wk, k, &rows, &ins, scalars, outs, &limits)? {
                    Some(direct) => Ok(direct),
                    None => rows_wg(
                        d,
                        wk,
                        k,
                        &rows,
                        &ins,
                        scalars,
                        outs,
                        &limits,
                        fits(dev, &ins, n, &out_elems, &limits),
                        budget(dev, &limits),
                    ),
                }
            },
            _ => Err(CoreError::Device(
                "kernel compilé pour un autre périphérique".into(),
            )),
        }
    } else {
        match (dev, &c.imp) {
            (Device::Cpu, Imp::Cpu) => unsafe { run_cpu(k, &ins, scalars, n, outs) },
            (Device::Cl(d), Imp::Cl(ck)) if d.info.unified_memory => unsafe {
                cl_direct(d, ck, k, &ins, &arrays, scalars, n, outs, &limits)
            },
            (Device::Cl(d), Imp::Cl(ck)) => {
                if fits(dev, &ins, n, &out_elems, &limits) {
                    unsafe { cl_resident(d, ck, k, &ins, scalars, n, &limits) }
                } else {
                    unsafe { cl_flux(d, ck, k, &ins, &arrays, scalars, n, outs, &limits) }
                }
            }
            (Device::Wg(d), Imp::Wg(wk)) => {
                let (words, fwords) = wg_params(k, scalars)?;
                if let Some(direct) = unsafe {
                    wg_direct(
                        d, wk, &ins, &arrays, &words, &fwords, n, outs, &out_elems, &limits,
                    )?
                } {
                    Ok(direct)
                } else if fits(dev, &ins, n, &out_elems, &limits) {
                    unsafe { wg_resident(d, wk, &ins, &words, &fwords, n, &out_elems, &limits) }
                } else {
                    let flux_in: Vec<FluxIn> = ins
                        .iter()
                        .zip(&arrays)
                        .map(|(i, ty)| match i {
                            In::Host(p, _) => Ok(FluxIn::Host {
                                ptr: *p,
                                elem: ty.size(),
                            }),
                            In::Dev(a) => match &a.storage {
                                Storage::Wg(b) => Ok(FluxIn::Dev {
                                    buf: b,
                                    elem: ty.size(),
                                }),
                                _ => Err(other_kind()),
                            },
                        })
                        .collect::<R<_>>()?;
                    let elem_max = elem_max(&arrays, &out_elems);
                    let max_buf = d.info.max_buffer_bytes.unwrap_or(u64::MAX) as usize;
                    let chunk_elems = (limits.chunk_bytes.min(max_buf) / elem_max).max(1);
                    let fouts: Vec<(*mut u8, usize)> = outs
                        .iter()
                        .copied()
                        .zip(out_elems.iter().copied())
                        .collect();
                    unsafe {
                        d.flux(wk, &flux_in, &fouts, &words, &fwords, n, chunk_elems)
                            .map_err(dev_err)?
                    };
                    Ok((
                        outs.iter().map(|p| Storage::Host { ptr: *p }).collect(),
                        Mode::Flux,
                        false,
                    ))
                }
            }
            _ => Err(CoreError::Device(
                "kernel compilé pour un autre périphérique".into(),
            )),
        }
    };
    let (storages, mode, mut host_backed_io) = result?;
    // Entrées rapatriées d'un autre périphérique : copies locales libérées au retour.
    if !staged.is_empty() && host_backed_io {
        dev.finish()?;
        host_backed_io = false;
    }
    Ok(storages
        .into_iter()
        .zip(&k.outputs)
        .map(|(storage, ty)| DeviceArray {
            device: dev.clone(),
            storage,
            dtype: *ty,
            len: n,
            mode,
            host_backed_io,
        })
        .collect())
}

/// Stockage de chaque sortie, mode retenu, buffers encore adossés à la mémoire hôte.
type Planned = R<(Vec<Storage>, Mode, bool)>;

fn other_kind() -> CoreError {
    CoreError::Device("entrée sur un autre type de périphérique".into())
}

fn elem_max(arrays: &[Scalar], out_elems: &[usize]) -> usize {
    arrays
        .iter()
        .map(|t| t.size())
        .chain(out_elems.iter().copied())
        .max()
        .unwrap_or(4)
}

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

fn budget(dev: &Device, limits: &TransferLimits) -> u64 {
    limits
        .budget_bytes
        .unwrap_or_else(|| device_memory(dev) / 2)
}

/// L'appel tient-il en mode Résident (budget, taille max d'un buffer, une seule grille) ?
fn fits(dev: &Device, ins: &[In], n: usize, out_elems: &[usize], limits: &TransferLimits) -> bool {
    let budget = limits.budget_bytes.unwrap_or(device_memory(dev) / 2);
    let max_buf = dev.info().max_buffer_bytes.unwrap_or(u64::MAX);
    let outs = out_elems.iter().map(|e| (n * e) as u64);
    let biggest = ins
        .iter()
        .map(|i| i.bytes() as u64)
        .chain(outs.clone())
        .max()
        .unwrap_or(0);
    let new_bytes: u64 = ins
        .iter()
        .filter(|i| matches!(i, In::Host(..)))
        .map(|i| i.bytes() as u64)
        .sum::<u64>()
        + outs.sum::<u64>();
    n <= u32::MAX as usize && new_bytes <= budget && biggest <= max_buf
}

/// Blocs `(début, longueur)` de la grille élément par élément.
fn blocks(n: usize, chunk: usize) -> Vec<(usize, usize)> {
    (0..n)
        .step_by(chunk.max(1))
        .map(|s| (s, chunk.min(n - s)))
        .collect()
}

/// Liste complète des scalaires (ordre des paramètres) : ceux de l'appelant, et des zéros à la
/// place des paramètres du moteur (remplis à chaque lancement).
fn full_scalars(k: &Kernel, user: &[Const]) -> Vec<Const> {
    let mut u = user.iter();
    k.params
        .iter()
        .enumerate()
        .filter(|(_, p)| p.kind == ParamKind::Scalar)
        .map(|(i, p)| {
            if k.is_engine_param(i) {
                match p.ty {
                    Scalar::U32 => Const::U32(0),
                    _ => Const::I32(0),
                }
            } else {
                *u.next().unwrap()
            }
        })
        .collect()
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

/// Arguments OpenCL dans l'ordre des paramètres : tableaux = `bufs` (dans l'ordre), puis les
/// sorties et n.
fn cl_args<'a>(
    k: &Kernel,
    bufs: &[&'a ClBuffer],
    scalars: &[Const],
    outs: &[&'a ClBuffer],
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
    args.extend(outs.iter().map(|o| ClArg::Buf(o)));
    args.push(ClArg::U32(n));
    args
}

unsafe fn run_cpu(
    k: &Kernel,
    ins: &[In],
    scalars: &[Const],
    n: usize,
    outs: &[*mut u8],
) -> Planned {
    let slices: Vec<&[u8]> = ins
        .iter()
        .map(|i| match i {
            In::Host(p, b) => Ok(unsafe { std::slice::from_raw_parts(*p, *b) }),
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
    let mut out_slices: Vec<&mut [u8]> = outs
        .iter()
        .zip(&k.outputs)
        .map(|(p, t)| unsafe { std::slice::from_raw_parts_mut(*p, n * t.size()) })
        .collect();
    interp::run(k, n, &args, &mut out_slices).map_err(CoreError::Device)?;
    Ok((
        outs.iter().map(|p| Storage::Host { ptr: *p }).collect(),
        Mode::Cpu,
        false,
    ))
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
    outs: &[*mut u8],
    limits: &TransferLimits,
) -> Planned {
    let out_elems: Vec<usize> = k.outputs.iter().map(|t| t.size()).collect();
    let max_alloc = d
        .info
        .max_buffer_bytes
        .unwrap_or(u64::MAX)
        .min(usize::MAX as u64) as usize;
    let chunk_bytes = limits
        .direct_chunk_bytes
        .unwrap_or(max_alloc)
        .min(max_alloc);
    let chunk = (chunk_bytes / elem_max(arrays, &out_elems)).clamp(1, u32::MAX as usize);
    let mut out_bufs: Vec<Vec<ClBuffer>> = outs.iter().map(|_| Vec::new()).collect();
    for (start, len) in blocks(n, chunk) {
        let mut bufs = Vec::new();
        for (i, ty) in ins.iter().zip(arrays) {
            let In::Host(p, _) = i else {
                return Err(CoreError::Device(
                    "mode Direct : entrée restée sur un périphérique".into(),
                ));
            };
            let (off, bytes) = (start * ty.size(), len * ty.size());
            bufs.push(unsafe { d.wrap_host(p.add(off) as *mut u8, bytes) }.map_err(dev_err)?);
        }
        let mut obs = Vec::new();
        for (p, e) in outs.iter().zip(&out_elems) {
            obs.push(unsafe { d.wrap_host(p.add(start * e), len * e) }.map_err(dev_err)?);
        }
        let refs: Vec<&ClBuffer> = bufs.iter().collect();
        let orefs: Vec<&ClBuffer> = obs.iter().collect();
        d.launch(
            ck,
            &cl_args(k, &refs, scalars, &orefs, len as u32),
            len,
            &[],
        )
        .map_err(dev_err)?;
        for (dst, ob) in out_bufs.iter_mut().zip(obs) {
            dst.push(ob);
        }
    }
    Ok((
        out_bufs
            .into_iter()
            .zip(outs)
            .map(|(bufs, ptr)| Storage::ClDirect { bufs, ptr: *ptr })
            .collect(),
        Mode::Direct,
        true,
    ))
}

/// Mode Direct sur Metal : chaque bloc est un buffer `NoCopy` sur la mémoire de l'hôte (aucune copie).
/// `None` si impossible (pas Metal, adresse non alignée sur 16 Kio, entrée restée sur un
/// périphérique, kernel avec index trop gros pour un seul buffer).
#[allow(clippy::too_many_arguments)]
unsafe fn wg_direct(
    d: &gpusync_backend_wgpu::WgDevice,
    wk: &gpusync_backend_wgpu::WgKernel,
    ins: &[In],
    arrays: &[Scalar],
    words: &[u32],
    fwords: &[f64],
    n: usize,
    outs: &[*mut u8],
    out_elems: &[usize],
    limits: &TransferLimits,
) -> R<Option<(Vec<Storage>, Mode, bool)>> {
    const PAGE: usize = 16384;
    if d.info.backend != "metal" || outs.iter().any(|p| !(*p as usize).is_multiple_of(PAGE)) {
        return Ok(None);
    }
    for i in ins {
        match i {
            In::Host(p, _) if (*p as usize).is_multiple_of(PAGE) => {}
            _ => return Ok(None),
        }
    }
    let max_buf = d
        .info
        .max_buffer_bytes
        .unwrap_or(u64::MAX)
        .min(usize::MAX as u64) as usize;
    let chunk_bytes = limits.direct_chunk_bytes.unwrap_or(max_buf).min(max_buf);
    let step = PAGE / 4; // débuts de blocs alignés sur une page pour des éléments de 4 ou 8 o
    let chunk = ((chunk_bytes / elem_max(arrays, out_elems)) / step * step).max(step);
    let refused = || CoreError::Device("Metal a refusé une partie de la mémoire hôte".into());
    let mut out_bufs: Vec<Vec<WgBuffer>> = outs.iter().map(|_| Vec::new()).collect();
    for (start, len) in blocks(n, chunk) {
        let mut bufs = Vec::new();
        for (i, ty) in ins.iter().zip(arrays) {
            let In::Host(p, _) = i else { unreachable!() };
            let (off, bytes) = (start * ty.size(), len * ty.size());
            match unsafe { d.wrap_host(p.add(off) as *mut u8, bytes) }.map_err(dev_err)? {
                Some(b) => bufs.push(b),
                None if start == 0 => return Ok(None),
                None => return Err(refused()),
            }
        }
        let mut obs = Vec::new();
        for (p, e) in outs.iter().zip(out_elems) {
            match unsafe { d.wrap_host(p.add(start * e), len * e) }.map_err(dev_err)? {
                Some(b) => obs.push(b),
                None if start == 0 => return Ok(None),
                None => return Err(refused()),
            }
        }
        let mut w = words.to_vec();
        w.push(len as u32);
        let refs: Vec<&WgBuffer> = bufs.iter().collect();
        let orefs: Vec<&WgBuffer> = obs.iter().collect();
        d.launch(wk, &refs, &orefs, &w, fwords, len)
            .map_err(dev_err)?;
        for (dst, ob) in out_bufs.iter_mut().zip(obs) {
            dst.push(ob);
        }
    }
    Ok(Some((
        out_bufs
            .into_iter()
            .zip(outs)
            .map(|(bufs, ptr)| Storage::WgDirect { bufs, ptr: *ptr })
            .collect(),
        Mode::Direct,
        true,
    )))
}

/// Mode Résident (wgpu) : entrées envoyées par blocs, sorties gardées sur le périphérique.
#[allow(clippy::too_many_arguments)]
unsafe fn wg_resident(
    d: &gpusync_backend_wgpu::WgDevice,
    wk: &gpusync_backend_wgpu::WgKernel,
    ins: &[In],
    words: &[u32],
    fwords: &[f64],
    n: usize,
    out_elems: &[usize],
    limits: &TransferLimits,
) -> Planned {
    let mut owned = Vec::new();
    for i in ins {
        if let In::Host(p, bytes) = i {
            let b = d.alloc(*bytes).map_err(dev_err)?;
            unsafe {
                d.upload(&b, 0, *p, *bytes, limits.chunk_bytes)
                    .map_err(dev_err)?
            };
            owned.push(b);
        }
    }
    let mut own = owned.iter();
    let bufs: Vec<&WgBuffer> = ins
        .iter()
        .map(|i| match i {
            In::Host(..) => Ok(own.next().unwrap()),
            In::Dev(a) => match &a.storage {
                Storage::Wg(b) => Ok(b),
                _ => Err(other_kind()),
            },
        })
        .collect::<R<_>>()?;
    let obufs: Vec<WgBuffer> = out_elems
        .iter()
        .map(|e| d.alloc(n * e).map_err(dev_err))
        .collect::<R<_>>()?;
    let orefs: Vec<&WgBuffer> = obufs.iter().collect();
    let mut w = words.to_vec();
    w.push(n as u32);
    d.launch(wk, &bufs, &orefs, &w, fwords, n)
        .map_err(dev_err)?;
    Ok((
        obufs.into_iter().map(Storage::Wg).collect(),
        Mode::Resident,
        false,
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
    if bytes == 0 {
        return Ok(None);
    }
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
    if bytes == 0 {
        return wait(done).map_err(dev_err);
    }
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

/// Mode Résident (OpenCL, GPU dédié) : entrées envoyées par l'anneau épinglé, sorties gardées en VRAM.
#[allow(clippy::too_many_arguments)]
unsafe fn cl_resident(
    d: &ClDevice,
    ck: &ClKernel,
    k: &Kernel,
    ins: &[In],
    scalars: &[Const],
    n: usize,
    limits: &TransferLimits,
) -> Planned {
    let mut owned = Vec::new();
    let mut last: Option<ClEvent> = None;
    for i in ins {
        if let In::Host(p, bytes) = i {
            let b = d.alloc(*bytes).map_err(dev_err)?;
            if let Some(e) = unsafe { cl_upload(d, &b, *p, *bytes, limits.chunk_bytes)? } {
                last = Some(e);
            }
            owned.push(b);
        }
    }
    let mut own = owned.iter();
    let bufs: Vec<&ClBuffer> = ins
        .iter()
        .map(|i| match i {
            In::Host(..) => Ok(own.next().unwrap()),
            In::Dev(a) => match &a.storage {
                Storage::Cl { buf, .. } => Ok(buf),
                _ => Err(other_kind()),
            },
        })
        .collect::<R<_>>()?;
    // Entrées d'un calcul précédent sur ce périphérique : attendre qu'elles soient prêtes.
    let mut waits: Vec<&ClEvent> = last.iter().collect();
    for i in ins {
        if let In::Dev(a) = i {
            if let Storage::Cl { done, .. } = &a.storage {
                waits.push(done);
            }
        }
    }
    let obufs: Vec<ClBuffer> = k
        .outputs
        .iter()
        .map(|t| d.alloc(n * t.size()).map_err(dev_err))
        .collect::<R<_>>()?;
    let orefs: Vec<&ClBuffer> = obufs.iter().collect();
    let done = Arc::new(
        d.launch(ck, &cl_args(k, &bufs, scalars, &orefs, n as u32), n, &waits)
            .map_err(dev_err)?,
    );
    Ok((
        obufs
            .into_iter()
            .map(|buf| Storage::Cl {
                buf,
                done: done.clone(),
            })
            .collect(),
        Mode::Resident,
        false,
    ))
}

/// Mode Flux (OpenCL, GPU dédié, kernels élément par élément) : pipeline à 3 emplacements sur deux
/// files ; pendant le calcul du bloc k, le bloc k+1 est envoyé et le bloc k−1 revient dans sa
/// destination finale.
#[allow(clippy::too_many_arguments)]
unsafe fn cl_flux(
    d: &ClDevice,
    ck: &ClKernel,
    k: &Kernel,
    ins: &[In],
    arrays: &[Scalar],
    scalars: &[Const],
    n: usize,
    outs: &[*mut u8],
    limits: &TransferLimits,
) -> Planned {
    const SLOTS: usize = 3;
    let out_elems: Vec<usize> = k.outputs.iter().map(|t| t.size()).collect();
    let max_alloc = d.info.max_buffer_bytes.unwrap_or(u64::MAX) as usize;
    let chunk = (limits.chunk_bytes.min(max_alloc) / elem_max(arrays, &out_elems)).clamp(1, n);
    struct Slot {
        ins: Vec<ClBuffer>,
        pins: Vec<Option<Pinned>>,
        outs: Vec<ClBuffer>,
        pin_outs: Vec<Pinned>,
        /// Bloc en cours : début, longueur, une lecture par sortie.
        busy: Option<(usize, usize, Vec<ClEvent>)>,
    }
    let mut ring: Vec<Slot> = Vec::new();
    for _ in 0..SLOTS.min(n.div_ceil(chunk)) {
        let mut sins = Vec::new();
        let mut pins = Vec::new();
        for (i, ty) in ins.iter().zip(arrays) {
            sins.push(d.alloc(chunk * ty.size()).map_err(dev_err)?);
            pins.push(match i {
                In::Host(..) => Some(d.pinned(chunk * ty.size()).map_err(dev_err)?),
                In::Dev(_) => None,
            });
        }
        let mut souts = Vec::new();
        let mut pouts = Vec::new();
        for e in &out_elems {
            souts.push(d.alloc(chunk * e).map_err(dev_err)?);
            pouts.push(d.pinned(chunk * e).map_err(dev_err)?);
        }
        ring.push(Slot {
            ins: sins,
            pins,
            outs: souts,
            pin_outs: pouts,
            busy: None,
        });
    }
    let finish = |s: &mut Slot| -> R<()> {
        if let Some((start, len, evs)) = s.busy.take() {
            for (o, e) in evs.iter().enumerate() {
                wait(e).map_err(dev_err)?;
                let sz = out_elems[o];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        s.pin_outs[o].ptr,
                        outs[o].add(start * sz),
                        len * sz,
                    )
                };
            }
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
                In::Host(p, _) => {
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
                    _ => return Err(other_kind()),
                },
            }
        }
        d.flush().map_err(dev_err)?;
        let refs: Vec<&ClBuffer> = s.ins.iter().collect();
        let orefs: Vec<&ClBuffer> = s.outs.iter().collect();
        let wait_w: Vec<&ClEvent> = writes.iter().collect();
        let ek = d
            .launch(
                ck,
                &cl_args(k, &refs, scalars, &orefs, len as u32),
                len,
                &wait_w,
            )
            .map_err(dev_err)?;
        let mut reads = Vec::new();
        for (o, e) in out_elems.iter().enumerate() {
            reads.push(
                unsafe { d.read_at(&s.outs[o], 0, s.pin_outs[o].ptr, len * e, &[&ek]) }
                    .map_err(dev_err)?,
            );
        }
        d.flush().map_err(dev_err)?;
        s.busy = Some((start, len, reads));
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
        for p in s.pin_outs {
            d.recycle(p);
        }
    }
    Ok((
        outs.iter().map(|p| Storage::Host { ptr: *p }).collect(),
        Mode::Flux,
        false,
    ))
}

// ───────────────────── Kernels avec index : blocs de lignes (§9.7) ─────────────────────

/// Plan par lignes d'un kernel avec index : la grille (forme de la sortie) est découpée le long de
/// son premier axe. Les entrées `chunked` (lues seulement à la ligne courante) suivent le même
/// découpage ; les autres restent entières sur le périphérique.
struct Rows {
    /// Lignes de la sortie (taille du premier axe).
    total: usize,
    /// Cases par ligne.
    rest: usize,
    /// Par entrée : découpée par lignes ?
    chunked: Vec<bool>,
    /// Octets par ligne : de chaque entrée découpée (0 sinon), de chaque sortie.
    in_row: Vec<usize>,
    out_row: Vec<usize>,
    /// Positions de row0, inrow, obase dans la liste des scalaires.
    slots: [usize; 3],
    /// Nom de chaque entrée (messages d'erreur).
    names: Vec<String>,
}

impl Rows {
    fn new(k: &Kernel, scalars: &[Const], n: usize, ins: &[In], out_elems: &[usize]) -> Rows {
        let Grid::Indexed {
            axes,
            chunked,
            row0,
            inrow,
            obase,
        } = &k.grid
        else {
            unreachable!()
        };
        let pos = |param: usize| {
            k.params[..param]
                .iter()
                .filter(|p| p.kind == ParamKind::Scalar)
                .count()
        };
        let total = match interp::convert(scalars[pos(axes[0])], Scalar::I32) {
            Const::I32(v) => v.max(1) as usize,
            _ => 1,
        };
        let rest = n / total;
        let arrays: Vec<(usize, String)> =
            k.array_params().map(|(i, p)| (i, p.name.clone())).collect();
        let is_chunked: Vec<bool> = arrays.iter().map(|(i, _)| chunked.contains(i)).collect();
        Rows {
            total,
            rest,
            in_row: ins
                .iter()
                .zip(&is_chunked)
                .map(|(i, c)| if *c { i.bytes() / total } else { 0 })
                .collect(),
            out_row: out_elems.iter().map(|e| rest * e).collect(),
            chunked: is_chunked,
            slots: [pos(*row0), pos(*inrow), pos(*obase)],
            names: arrays.into_iter().map(|(_, n)| n).collect(),
        }
    }

    /// Scalaires d'un lancement : lignes `row0…`, entrées découpées à partir de la ligne `inrow`,
    /// sorties écrites à partir de la case `obase` de leur buffer.
    fn with(&self, scalars: &[Const], row0: usize, inrow: usize, obase: usize) -> Vec<Const> {
        let mut s = scalars.to_vec();
        s[self.slots[0]] = Const::I32(row0 as i32);
        s[self.slots[1]] = Const::I32(inrow as i32);
        s[self.slots[2]] = Const::U32(obase as u32);
        s
    }

    /// Plus grand nombre d'octets par ligne parmi les tableaux découpés (entrées et sorties).
    fn max_row(&self) -> usize {
        self.in_row
            .iter()
            .chain(&self.out_row)
            .copied()
            .max()
            .unwrap_or(1)
            .max(1)
    }

    /// Octets par ligne, tous tableaux découpés confondus.
    fn sum_row(&self) -> usize {
        (self.in_row.iter().sum::<usize>() + self.out_row.iter().sum::<usize>()).max(1)
    }

    /// Blocs de lignes `(début, lignes)` de `rows` lignes au plus.
    fn blocks(&self, rows: usize) -> Vec<(usize, usize)> {
        let rows = rows.clamp(1, self.total);
        (0..self.total)
            .step_by(rows)
            .map(|s| (s, rows.min(self.total - s)))
            .collect()
    }

    fn resident_too_big(
        &self,
        k: &Kernel,
        i: usize,
        bytes: usize,
        room: u64,
        dev: &str,
    ) -> CoreError {
        CoreError::Unsupported(format!(
            "'{}' : l'entrée '{}' ({}) est lue à n'importe quelle ligne de la sortie, elle doit \
             donc tenir entière sur {dev} ({} disponibles). Pour qu'elle soit découpée, indexez-la \
             par la ligne courante ({}[i, …] avec i, … = gs.index()) ; sinon augmentez gs.set_memory_budget() \
             ou utilisez un autre périphérique",
            k.name,
            self.names[i],
            human(bytes as u64),
            human(room),
            self.names[i],
        ))
    }
}

/// Taille lisible : « 1,50 Gio », « 12,0 Mio », « 300 o ».
fn human(b: u64) -> String {
    let f = b as f64;
    let s = if b >= 1 << 30 {
        format!("{:.2} Gio", f / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} Mio", f / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} Kio", f / 1024.0)
    } else {
        format!("{b} o")
    };
    s.replace('.', ",")
}

/// Cases lancées sans mesure (appel court) ou lors de la première mesure.
const PROBE_CELLS: usize = 1 << 16;
/// Durée visée d'un lancement : bien en deçà du délai de garde des drivers (TDR Windows : ~2 s).
const TARGET_LAUNCH_S: f64 = 0.1;
/// Cases au plus par lancement (indices u32, limites de la grille wgpu).
const MAX_LAUNCH_CELLS: usize = 1 << 30;

/// Découpe les lancements d'un kernel avec index pour qu'aucun ne dure plus de ~0,1 s : le premier
/// est petit et chronométré, les suivants sont dimensionnés d'après le débit mesuré.
#[derive(Default)]
struct Pacer {
    /// Débit mesuré (lignes par seconde).
    rate: Option<f64>,
    last: usize,
}

impl Pacer {
    /// Calcule les lignes `[a, b)` ; `step(première ligne, lignes, attendre la fin)`.
    fn run(
        &mut self,
        a: usize,
        b: usize,
        rest: usize,
        step: &mut dyn FnMut(usize, usize, bool) -> R<()>,
    ) -> R<()> {
        let rest = rest.max(1);
        let max_rows = (MAX_LAUNCH_CELLS / rest).max(1);
        if self.rate.is_none() && (b - a) * rest <= PROBE_CELLS {
            return step(a, b - a, false);
        }
        let mut s = a;
        while s < b {
            let want = match self.rate {
                None => (PROBE_CELLS / rest).max(1),
                Some(r) => ((r * TARGET_LAUNCH_S) as usize).clamp(1, self.last.saturating_mul(8)),
            };
            let len = want.min(max_rows).min(b - s);
            let last = s + len == b;
            let t0 = std::time::Instant::now();
            step(s, len, !last)?;
            if !last {
                let dt = t0.elapsed().as_secs_f64().max(1e-6);
                self.rate = Some(len as f64 / dt);
                self.last = len;
            }
            s += len;
        }
        Ok(())
    }
}

fn max_buffer(info_max: Option<u64>) -> usize {
    info_max.unwrap_or(u64::MAX).min(usize::MAX as u64) as usize
}

/// Kernel avec index, OpenCL sur mémoire unifiée : chaque bloc de lignes utilise directement la
/// mémoire de l'hôte ; les entrées lues à toutes les lignes sont importées une fois.
#[allow(clippy::too_many_arguments)]
unsafe fn rows_cl_direct(
    d: &ClDevice,
    ck: &ClKernel,
    k: &Kernel,
    rows: &Rows,
    ins: &[In],
    scalars: &[Const],
    outs: &[*mut u8],
    limits: &TransferLimits,
) -> Planned {
    let max_alloc = max_buffer(d.info.max_buffer_bytes);
    let cap = limits
        .direct_chunk_bytes
        .unwrap_or(max_alloc)
        .min(max_alloc);
    let mut whole: Vec<Option<ClBuffer>> = Vec::new();
    for (i, input) in ins.iter().enumerate() {
        let In::Host(p, bytes) = input else {
            return Err(CoreError::Device(
                "mode Direct : entrée restée sur un périphérique".into(),
            ));
        };
        whole.push(if rows.chunked[i] {
            None
        } else {
            if *bytes > max_alloc {
                return Err(rows.resident_too_big(k, i, *bytes, max_alloc as u64, &d.info.name));
            }
            Some(unsafe { d.wrap_host(*p as *mut u8, *bytes) }.map_err(dev_err)?)
        });
    }
    let mut pacer = Pacer::default();
    let mut out_bufs: Vec<Vec<ClBuffer>> = outs.iter().map(|_| Vec::new()).collect();
    for (r0, len) in rows.blocks(cap / rows.max_row()) {
        let mut parts: Vec<Option<ClBuffer>> = Vec::new();
        for (i, input) in ins.iter().enumerate() {
            parts.push(match input {
                In::Host(p, _) if rows.chunked[i] => {
                    let rb = rows.in_row[i];
                    Some(
                        unsafe { d.wrap_host(p.add(r0 * rb) as *mut u8, len * rb) }
                            .map_err(dev_err)?,
                    )
                }
                _ => None,
            });
        }
        let mut obs = Vec::new();
        for (p, rb) in outs.iter().zip(&rows.out_row) {
            obs.push(unsafe { d.wrap_host(p.add(r0 * rb), len * rb) }.map_err(dev_err)?);
        }
        let refs: Vec<&ClBuffer> = parts
            .iter()
            .zip(&whole)
            .map(|(a, b)| a.as_ref().or(b.as_ref()).unwrap())
            .collect();
        let orefs: Vec<&ClBuffer> = obs.iter().collect();
        pacer.run(r0, r0 + len, rows.rest, &mut |s, l, sync| {
            let sc = rows.with(scalars, s, r0, (s - r0) * rows.rest);
            let cells = l * rows.rest;
            let e = d
                .launch(
                    ck,
                    &cl_args(k, &refs, &sc, &orefs, cells as u32),
                    cells,
                    &[],
                )
                .map_err(dev_err)?;
            if sync {
                wait(&e).map_err(dev_err)?;
            }
            Ok(())
        })?;
        for (dst, ob) in out_bufs.iter_mut().zip(obs) {
            dst.push(ob);
        }
        // Les buffers d'entrée du bloc peuvent être libérés : la file garde ses références.
    }
    // Les imports entiers doivent survivre aux calculs : ils sont rattachés à la première sortie.
    let mut storages: Vec<Storage> = out_bufs
        .into_iter()
        .zip(outs)
        .map(|(bufs, ptr)| Storage::ClDirect { bufs, ptr: *ptr })
        .collect();
    if let Some(Storage::ClDirect { bufs, .. }) = storages.first_mut() {
        bufs.extend(whole.into_iter().flatten());
    }
    Ok((storages, Mode::Direct, true))
}

/// Kernel avec index, OpenCL sur GPU dédié : Résident (tout tient) ou par blocs de lignes, avec
/// deux emplacements pour recouvrir le retour d'un bloc par le calcul du suivant.
#[allow(clippy::too_many_arguments)]
unsafe fn rows_cl(
    d: &ClDevice,
    ck: &ClKernel,
    k: &Kernel,
    rows: &Rows,
    ins: &[In],
    scalars: &[Const],
    outs: &[*mut u8],
    limits: &TransferLimits,
    fits: bool,
    budget: u64,
) -> Planned {
    let max_buf = max_buffer(d.info.max_buffer_bytes);
    // Entrées entières (toutes si l'appel tient, sinon celles lues à n'importe quelle ligne).
    let mut owned: Vec<Option<ClBuffer>> = Vec::new();
    let mut waits_owned: Vec<ClEvent> = Vec::new();
    let mut resident_bytes = 0u64;
    for (i, input) in ins.iter().enumerate() {
        let keep_whole = fits || !rows.chunked[i];
        owned.push(match input {
            In::Host(p, bytes) if keep_whole => {
                resident_bytes += *bytes as u64;
                if *bytes > max_buf || (!fits && resident_bytes > budget) {
                    return Err(rows.resident_too_big(
                        k,
                        i,
                        *bytes,
                        budget.min(max_buf as u64),
                        &d.info.name,
                    ));
                }
                let b = d.alloc(*bytes).map_err(dev_err)?;
                if let Some(e) = unsafe { cl_upload(d, &b, *p, *bytes, limits.chunk_bytes)? } {
                    waits_owned.push(e);
                }
                Some(b)
            }
            _ => None,
        });
    }
    let dev_buf = |i: usize| -> R<(&ClBuffer, &ClEvent)> {
        match &ins[i] {
            In::Dev(a) => match &a.storage {
                Storage::Cl { buf, done } => Ok((buf, &**done)),
                _ => Err(other_kind()),
            },
            In::Host(..) => unreachable!(),
        }
    };
    let mut pacer = Pacer::default();

    if fits {
        let mut refs: Vec<&ClBuffer> = Vec::new();
        let mut waits: Vec<&ClEvent> = waits_owned.iter().collect();
        for (i, o) in owned.iter().enumerate() {
            match o {
                Some(b) => refs.push(b),
                None => {
                    let (b, done) = dev_buf(i)?;
                    refs.push(b);
                    waits.push(done);
                }
            }
        }
        let obufs: Vec<ClBuffer> = rows
            .out_row
            .iter()
            .map(|rb| d.alloc(rows.total * rb).map_err(dev_err))
            .collect::<R<_>>()?;
        let orefs: Vec<&ClBuffer> = obufs.iter().collect();
        let mut last: Option<ClEvent> = None;
        pacer.run(0, rows.total, rows.rest, &mut |s, l, sync| {
            let sc = rows.with(scalars, s, 0, s * rows.rest);
            let cells = l * rows.rest;
            let w: &[&ClEvent] = if last.is_none() { &waits } else { &[] };
            let e = d
                .launch(ck, &cl_args(k, &refs, &sc, &orefs, cells as u32), cells, w)
                .map_err(dev_err)?;
            if sync {
                wait(&e).map_err(dev_err)?;
            }
            last = Some(e);
            Ok(())
        })?;
        let done = Arc::new(last.unwrap());
        return Ok((
            obufs
                .into_iter()
                .map(|buf| Storage::Cl {
                    buf,
                    done: done.clone(),
                })
                .collect(),
            Mode::Resident,
            false,
        ));
    }

    // Par blocs de lignes : deux emplacements, chacun avec ses buffers d'entrées découpées et de sorties.
    let room = budget.saturating_sub(resident_bytes) as usize;
    let per_row: Vec<usize> = rows.in_row.iter().chain(&rows.out_row).copied().collect();
    let by_buf = per_row
        .iter()
        .filter(|b| **b > 0)
        .map(|b| max_buf / b)
        .min()
        .unwrap_or(rows.total);
    let br = (limits.chunk_bytes / rows.max_row())
        .min(room / (2 * rows.sum_row()))
        .min(by_buf)
        .clamp(1, rows.total);
    struct Slot {
        ins: Vec<Option<ClBuffer>>,
        outs: Vec<ClBuffer>,
        pending: Option<(usize, usize, ClEvent)>,
    }
    let mut slots: Vec<Slot> = Vec::new();
    for _ in 0..2.min(rows.total.div_ceil(br)) {
        slots.push(Slot {
            ins: rows
                .in_row
                .iter()
                .zip(&rows.chunked)
                .map(|(rb, c)| c.then(|| d.alloc(br * rb)).transpose().map_err(dev_err))
                .collect::<R<_>>()?,
            outs: rows
                .out_row
                .iter()
                .map(|rb| d.alloc(br * rb).map_err(dev_err))
                .collect::<R<_>>()?,
            pending: None,
        });
    }
    let drain = |s: &mut Slot| -> R<()> {
        if let Some((r0, len, done)) = s.pending.take() {
            for (o, buf) in s.outs.iter().enumerate() {
                let rb = rows.out_row[o];
                unsafe {
                    cl_download(
                        d,
                        buf,
                        &done,
                        outs[o].add(r0 * rb),
                        len * rb,
                        limits.chunk_bytes,
                    )?
                };
            }
        }
        Ok(())
    };
    for (b, (r0, len)) in rows.blocks(br).into_iter().enumerate() {
        let cur = b % slots.len();
        let mut writes: Vec<ClEvent> = Vec::new();
        for (i, input) in ins.iter().enumerate() {
            if !rows.chunked[i] {
                continue;
            }
            let rb = rows.in_row[i];
            let dst = slots[cur].ins[i].as_ref().unwrap();
            match input {
                In::Host(p, _) => {
                    if let Some(e) =
                        unsafe { cl_upload(d, dst, p.add(r0 * rb), len * rb, limits.chunk_bytes)? }
                    {
                        writes.push(e);
                    }
                }
                In::Dev(_) => {
                    let (src, done) = dev_buf(i)?;
                    writes.push(
                        d.copy_at(src, r0 * rb, dst, 0, len * rb, &[done])
                            .map_err(dev_err)?,
                    );
                }
            }
        }
        d.flush().map_err(dev_err)?;
        let mut waits: Vec<&ClEvent> = writes.iter().chain(&waits_owned).collect();
        for i in 0..ins.len() {
            if !rows.chunked[i] && owned[i].is_none() {
                waits.push(dev_buf(i)?.1);
            }
        }
        let refs: Vec<&ClBuffer> = (0..ins.len())
            .map(|i| match (&slots[cur].ins[i], &owned[i]) {
                (Some(b), _) | (None, Some(b)) => Ok(b),
                (None, None) => dev_buf(i).map(|(b, _)| b),
            })
            .collect::<R<_>>()?;
        let orefs: Vec<&ClBuffer> = slots[cur].outs.iter().collect();
        let mut last: Option<ClEvent> = None;
        pacer.run(r0, r0 + len, rows.rest, &mut |s, l, sync| {
            let sc = rows.with(scalars, s, r0, (s - r0) * rows.rest);
            let cells = l * rows.rest;
            let w: &[&ClEvent] = if last.is_none() { &waits } else { &[] };
            let e = d
                .launch(ck, &cl_args(k, &refs, &sc, &orefs, cells as u32), cells, w)
                .map_err(dev_err)?;
            d.flush().map_err(dev_err)?;
            if sync {
                wait(&e).map_err(dev_err)?;
            }
            last = Some(e);
            Ok(())
        })?;
        drop(refs);
        drop(orefs);
        // Pendant ce calcul, le bloc précédent revient dans sa destination.
        let prev = (cur + slots.len() - 1) % slots.len();
        if prev != cur {
            drain(&mut slots[prev])?;
        }
        slots[cur].pending = Some((r0, len, last.unwrap()));
    }
    for s in &mut slots {
        drain(s)?;
    }
    Ok((
        outs.iter().map(|p| Storage::Host { ptr: *p }).collect(),
        Mode::Flux,
        false,
    ))
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Kernel avec index sur Metal : blocs de lignes en `NoCopy` sur la mémoire de l'hôte (débuts de
/// blocs alignés sur une page). `None` si impossible (pas Metal, adresses non alignées…).
#[allow(clippy::too_many_arguments)]
unsafe fn rows_wg_direct(
    d: &gpusync_backend_wgpu::WgDevice,
    wk: &gpusync_backend_wgpu::WgKernel,
    k: &Kernel,
    rows: &Rows,
    ins: &[In],
    scalars: &[Const],
    outs: &[*mut u8],
    limits: &TransferLimits,
) -> R<Option<(Vec<Storage>, Mode, bool)>> {
    const PAGE: usize = 16384;
    if d.info.backend != "metal" || outs.iter().any(|p| !(*p as usize).is_multiple_of(PAGE)) {
        return Ok(None);
    }
    let max_buf = max_buffer(d.info.max_buffer_bytes);
    for i in ins {
        match i {
            In::Host(p, b) if (*p as usize).is_multiple_of(PAGE) && *b <= max_buf => {}
            _ => return Ok(None),
        }
    }
    let cap = limits.direct_chunk_bytes.unwrap_or(max_buf).min(max_buf);
    // Lignes par bloc : multiple de `step` pour que chaque bloc commence sur une page.
    let step = rows
        .in_row
        .iter()
        .chain(&rows.out_row)
        .filter(|b| **b > 0)
        .map(|b| PAGE / gcd(PAGE, *b))
        .fold(1, |acc, s| acc / gcd(acc, s) * s);
    let mut br = (cap / rows.max_row()).min(rows.total);
    if br < rows.total {
        br = br / step * step;
        if br == 0 {
            return Ok(None);
        }
    }
    let mut whole: Vec<Option<WgBuffer>> = Vec::new();
    for (i, input) in ins.iter().enumerate() {
        let In::Host(p, bytes) = input else {
            unreachable!()
        };
        whole.push(if rows.chunked[i] {
            None
        } else {
            match unsafe { d.wrap_host(*p as *mut u8, *bytes) }.map_err(dev_err)? {
                Some(b) => Some(b),
                None => return Ok(None),
            }
        });
    }
    let refused = || CoreError::Device("Metal a refusé une partie de la mémoire hôte".into());
    let mut pacer = Pacer::default();
    let mut out_bufs: Vec<Vec<WgBuffer>> = outs.iter().map(|_| Vec::new()).collect();
    let mut keep: Vec<WgBuffer> = Vec::new();
    for (r0, len) in rows.blocks(br) {
        let mut parts: Vec<Option<WgBuffer>> = Vec::new();
        for (i, input) in ins.iter().enumerate() {
            let In::Host(p, _) = input else {
                unreachable!()
            };
            parts.push(if rows.chunked[i] {
                let rb = rows.in_row[i];
                match unsafe { d.wrap_host(p.add(r0 * rb) as *mut u8, len * rb) }
                    .map_err(dev_err)?
                {
                    Some(b) => Some(b),
                    None if r0 == 0 => return Ok(None),
                    None => return Err(refused()),
                }
            } else {
                None
            });
        }
        let mut obs = Vec::new();
        for (p, rb) in outs.iter().zip(&rows.out_row) {
            match unsafe { d.wrap_host(p.add(r0 * rb), len * rb) }.map_err(dev_err)? {
                Some(b) => obs.push(b),
                None if r0 == 0 => return Ok(None),
                None => return Err(refused()),
            }
        }
        {
            let refs: Vec<&WgBuffer> = parts
                .iter()
                .zip(&whole)
                .map(|(a, b)| a.as_ref().or(b.as_ref()).unwrap())
                .collect();
            let orefs: Vec<&WgBuffer> = obs.iter().collect();
            pacer.run(r0, r0 + len, rows.rest, &mut |s, l, sync| {
                let sc = rows.with(scalars, s, r0, (s - r0) * rows.rest);
                let (mut w, f) = wg_params(k, &sc)?;
                let cells = l * rows.rest;
                w.push(cells as u32);
                d.launch(wk, &refs, &orefs, &w, &f, cells)
                    .map_err(dev_err)?;
                if sync {
                    d.finish().map_err(dev_err)?;
                }
                Ok(())
            })?;
        }
        keep.extend(parts.into_iter().flatten());
        for (dst, ob) in out_bufs.iter_mut().zip(obs) {
            dst.push(ob);
        }
    }
    let mut storages: Vec<Storage> = out_bufs
        .into_iter()
        .zip(outs)
        .map(|(bufs, ptr)| Storage::WgDirect { bufs, ptr: *ptr })
        .collect();
    if let Some(Storage::WgDirect { bufs, .. }) = storages.first_mut() {
        bufs.extend(keep);
        bufs.extend(whole.into_iter().flatten());
    }
    Ok(Some((storages, Mode::Direct, true)))
}

/// Kernel avec index via wgpu : Résident (tout tient) ou par blocs de lignes (une seule file :
/// envoi, calcul et retour de chaque bloc se suivent).
#[allow(clippy::too_many_arguments)]
unsafe fn rows_wg(
    d: &gpusync_backend_wgpu::WgDevice,
    wk: &gpusync_backend_wgpu::WgKernel,
    k: &Kernel,
    rows: &Rows,
    ins: &[In],
    scalars: &[Const],
    outs: &[*mut u8],
    limits: &TransferLimits,
    fits: bool,
    budget: u64,
) -> Planned {
    let max_buf = max_buffer(d.info.max_buffer_bytes);
    let mut owned: Vec<Option<WgBuffer>> = Vec::new();
    let mut resident_bytes = 0u64;
    for (i, input) in ins.iter().enumerate() {
        let keep_whole = fits || !rows.chunked[i];
        owned.push(match input {
            In::Host(p, bytes) if keep_whole => {
                resident_bytes += *bytes as u64;
                if *bytes > max_buf || (!fits && resident_bytes > budget) {
                    return Err(rows.resident_too_big(
                        k,
                        i,
                        *bytes,
                        budget.min(max_buf as u64),
                        &d.info.name,
                    ));
                }
                let b = d.alloc(*bytes).map_err(dev_err)?;
                unsafe {
                    d.upload(&b, 0, *p, *bytes, limits.chunk_bytes)
                        .map_err(dev_err)?
                };
                Some(b)
            }
            _ => None,
        });
    }
    let dev_buf = |i: usize| -> R<&WgBuffer> {
        match &ins[i] {
            In::Dev(a) => match &a.storage {
                Storage::Wg(b) => Ok(b),
                _ => Err(other_kind()),
            },
            In::Host(..) => unreachable!(),
        }
    };
    let mut pacer = Pacer::default();
    let mut launch = |refs: &[&WgBuffer],
                      orefs: &[&WgBuffer],
                      r0: usize,
                      a: usize,
                      b: usize,
                      whole_out: bool| {
        pacer.run(a, b, rows.rest, &mut |s, l, sync| {
            let obase = if whole_out {
                s * rows.rest
            } else {
                (s - r0) * rows.rest
            };
            let sc = rows.with(scalars, s, r0, obase);
            let (mut w, f) = wg_params(k, &sc)?;
            let cells = l * rows.rest;
            w.push(cells as u32);
            d.launch(wk, refs, orefs, &w, &f, cells).map_err(dev_err)?;
            if sync {
                d.finish().map_err(dev_err)?;
            }
            Ok(())
        })
    };

    if fits {
        let refs: Vec<&WgBuffer> = (0..ins.len())
            .map(|i| match &owned[i] {
                Some(b) => Ok(b),
                None => dev_buf(i),
            })
            .collect::<R<_>>()?;
        let obufs: Vec<WgBuffer> = rows
            .out_row
            .iter()
            .map(|rb| d.alloc(rows.total * rb).map_err(dev_err))
            .collect::<R<_>>()?;
        let orefs: Vec<&WgBuffer> = obufs.iter().collect();
        launch(&refs, &orefs, 0, 0, rows.total, true)?;
        drop(orefs);
        return Ok((
            obufs.into_iter().map(Storage::Wg).collect(),
            Mode::Resident,
            false,
        ));
    }

    let room = budget.saturating_sub(resident_bytes) as usize;
    let by_buf = rows
        .in_row
        .iter()
        .chain(&rows.out_row)
        .filter(|b| **b > 0)
        .map(|b| max_buf / b)
        .min()
        .unwrap_or(rows.total);
    let br = (limits.chunk_bytes / rows.max_row())
        .min(room / rows.sum_row())
        .min(by_buf)
        .clamp(1, rows.total);
    let sins: Vec<Option<WgBuffer>> = rows
        .in_row
        .iter()
        .zip(&rows.chunked)
        .map(|(rb, c)| c.then(|| d.alloc(br * rb)).transpose().map_err(dev_err))
        .collect::<R<_>>()?;
    let souts: Vec<WgBuffer> = rows
        .out_row
        .iter()
        .map(|rb| d.alloc(br * rb).map_err(dev_err))
        .collect::<R<_>>()?;
    for (r0, len) in rows.blocks(br) {
        for (i, input) in ins.iter().enumerate() {
            if !rows.chunked[i] {
                continue;
            }
            let rb = rows.in_row[i];
            let dst = sins[i].as_ref().unwrap();
            match input {
                In::Host(p, _) => unsafe {
                    d.upload(dst, 0, p.add(r0 * rb), len * rb, limits.chunk_bytes)
                        .map_err(dev_err)?
                },
                In::Dev(_) => d
                    .copy(dev_buf(i)?, r0 * rb, dst, 0, len * rb)
                    .map_err(dev_err)?,
            }
        }
        let refs: Vec<&WgBuffer> = (0..ins.len())
            .map(|i| match (&sins[i], &owned[i]) {
                (Some(b), _) | (None, Some(b)) => Ok(b),
                (None, None) => dev_buf(i),
            })
            .collect::<R<_>>()?;
        let orefs: Vec<&WgBuffer> = souts.iter().collect();
        launch(&refs, &orefs, r0, r0, r0 + len, false)?;
        for (o, buf) in souts.iter().enumerate() {
            let rb = rows.out_row[o];
            unsafe {
                d.download(buf, 0, outs[o].add(r0 * rb), len * rb, limits.chunk_bytes)
                    .map_err(dev_err)?
            };
        }
    }
    Ok((
        outs.iter().map(|p| Storage::Host { ptr: *p }).collect(),
        Mode::Flux,
        false,
    ))
}
