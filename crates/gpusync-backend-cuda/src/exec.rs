//! Exécution CUDA (ARCHITECTURE.md §8, §9), sur le modèle d'OpenCL :
//!
//! - **Deux streams** par périphérique : `compute` (kernels) et `transfer` (copies), chaînés par
//!   événements : les transferts d'un bloc se recouvrent avec le calcul d'un autre.
//! - **Mémoire épinglée** (`cuMemHostAlloc`) : les copies depuis (ou vers) ces tampons se font par DMA
//!   à pleine vitesse, et sans bloquer.
//! - **Compilation** : CUDA C → PTX par NVRTC (pour l'architecture du GPU, ou la plus récente connue de
//!   NVRTC), puis PTX → code machine par le driver.
//!
//! Le contexte principal du périphérique est rendu courant au début de chaque opération : les appels
//! peuvent venir de n'importe quel thread.

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::{Arc, Mutex, OnceLock};

use gpusync_runtime::{BackendError, DeviceInfo};

use crate::sys::{self, CUcontext, CUdeviceptr, CUevent, CUfunction, CUmodule, CUstream, Driver};

const BLOCK: u32 = 256;
/// Buffers mis en réserve pour être réutilisés : jusqu'à 64 Mio chacun, 512 Mio en tout.
const POOL_MAX_BUF: usize = 64 << 20;
const POOL_MAX_TOTAL: usize = 512 << 20;

/// Buffer en réserve : capacité, adresse, un événement par stream.
type Spare = (usize, CUdeviceptr, [CUevent; 2]);

/// Réserve de buffers libérés. `cuMemAlloc` / `cuMemFree` coûtent cher et `cuMemFree` synchronise
/// tout le GPU : un buffer rendu est gardé, avec un événement par stream, et n'est réutilisé
/// qu'une fois ces événements passés (plus aucune opération en file ne l'utilise).
struct BufPool {
    d: &'static Driver,
    compute: CUstream,
    transfer: CUstream,
    /// Buffers rendus (capacité, adresse, événements) et total des capacités.
    free: Mutex<(Vec<Spare>, usize)>,
}

unsafe impl Send for BufPool {}
unsafe impl Sync for BufPool {}

impl BufPool {
    fn take(&self, cap: usize) -> Option<CUdeviceptr> {
        let mut g = self.free.lock().unwrap_or_else(|e| e.into_inner());
        let pos = g.0.iter().position(|(c, _, evs)| {
            *c == cap
                && evs
                    .iter()
                    .all(|e| unsafe { (self.d.cuEventQuery)(*e) } == 0)
        })?;
        let (c, ptr, evs) = g.0.swap_remove(pos);
        g.1 -= c;
        for e in evs {
            unsafe { (self.d.cuEventDestroy_v2)(e) };
        }
        Some(ptr)
    }

    /// Garde le buffer s'il y a de la place ; sinon le libère.
    fn give(&self, cap: usize, ptr: CUdeviceptr) {
        let mut g = self.free.lock().unwrap_or_else(|e| e.into_inner());
        if g.1 + cap <= POOL_MAX_TOTAL {
            let mut evs = [std::ptr::null_mut(); 2];
            let ok = [self.compute, self.transfer]
                .iter()
                .zip(evs.iter_mut())
                .all(|(s, e)| unsafe {
                    (self.d.cuEventCreate)(e, sys::EVENT_DISABLE_TIMING) == 0
                        && (self.d.cuEventRecord)(*e, *s) == 0
                });
            if ok {
                g.0.push((cap, ptr, evs));
                g.1 += cap;
                return;
            }
            for e in evs.into_iter().filter(|e| !e.is_null()) {
                unsafe { (self.d.cuEventDestroy_v2)(e) };
            }
        }
        drop(g);
        unsafe { (self.d.cuMemFree_v2)(ptr) };
    }
}

/// Taille réservée pour `bytes` : puissance de deux (réutilisation fréquente) jusqu'à 64 Mio,
/// taille exacte au-delà (jamais mis en réserve).
fn capacity(bytes: usize) -> usize {
    let b = bytes.max(256);
    if b <= POOL_MAX_BUF {
        b.next_power_of_two()
    } else {
        b
    }
}

/// Un périphérique CUDA ouvert.
pub struct CuDevice {
    pub info: DeviceInfo,
    d: &'static Driver,
    ctx: CUcontext,
    compute: CUstream,
    transfer: CUstream,
    /// Capacité de calcul (ex. 89 pour 8.9).
    cc: i32,
    pool: Arc<BufPool>,
    kernels: Mutex<HashMap<String, Arc<CuKernel>>>,
    pinned_pool: Mutex<HashMap<usize, Vec<Pinned>>>,
}

// Les objets du driver CUDA sont utilisables depuis tous les threads (contexte rendu courant).
unsafe impl Send for CuDevice {}
unsafe impl Sync for CuDevice {}

/// Kernel compilé (module PTX chargé et sa fonction).
pub struct CuKernel {
    d: &'static Driver,
    ctx: CUcontext,
    module: CUmodule,
    func: CUfunction,
}

unsafe impl Send for CuKernel {}
unsafe impl Sync for CuKernel {}

impl Drop for CuKernel {
    fn drop(&mut self) {
        unsafe {
            (self.d.cuCtxSetCurrent)(self.ctx);
            (self.d.cuModuleUnload)(self.module);
        }
    }
}

/// Buffer en mémoire du périphérique.
pub struct CuBuffer {
    d: &'static Driver,
    ctx: CUcontext,
    pool: Arc<BufPool>,
    cap: usize,
    pub ptr: CUdeviceptr,
    pub bytes: usize,
}

unsafe impl Send for CuBuffer {}
unsafe impl Sync for CuBuffer {}

impl Drop for CuBuffer {
    fn drop(&mut self) {
        unsafe { (self.d.cuCtxSetCurrent)(self.ctx) };
        if self.cap <= POOL_MAX_BUF {
            self.pool.give(self.cap, self.ptr);
        } else {
            unsafe { (self.d.cuMemFree_v2)(self.ptr) };
        }
    }
}

/// Événement : fin d'une opération mise en file.
pub struct CuEvent {
    d: &'static Driver,
    ev: CUevent,
}

unsafe impl Send for CuEvent {}
unsafe impl Sync for CuEvent {}

impl Drop for CuEvent {
    fn drop(&mut self) {
        unsafe { (self.d.cuEventDestroy_v2)(self.ev) };
    }
}

impl CuEvent {
    pub fn wait(&self) -> Result<(), BackendError> {
        self.d
            .check(unsafe { (self.d.cuEventSynchronize)(self.ev) }, "attente")
    }
}

/// Tampon hôte épinglé ; rendu au pool par `recycle`, libéré à la destruction.
pub struct Pinned {
    d: &'static Driver,
    ctx: CUcontext,
    pub ptr: *mut u8,
    pub bytes: usize,
}

unsafe impl Send for Pinned {}
unsafe impl Sync for Pinned {}

impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe {
            (self.d.cuCtxSetCurrent)(self.ctx);
            (self.d.cuMemFreeHost)(self.ptr.cast());
        }
    }
}

/// Argument de lancement, dans l'ordre des paramètres du kernel.
pub enum CuArg<'a> {
    Buf(&'a CuBuffer),
    I32(i32),
    U32(u32),
    F32(f32),
    F64(f64),
    I64(i64),
    U64(u64),
}

static OPEN: OnceLock<Mutex<HashMap<usize, Arc<CuDevice>>>> = OnceLock::new();

impl CuDevice {
    /// Ouvre (une seule fois par processus) le périphérique `cuda:<index>`.
    pub fn open(index: usize) -> Result<Arc<CuDevice>, BackendError> {
        let mut open = OPEN
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(d) = open.get(&index) {
            return Ok(d.clone());
        }
        let (dev, info) = crate::enumerate()?
            .into_iter()
            .nth(index)
            .ok_or_else(|| BackendError(format!("périphérique cuda:{index} introuvable")))?;
        let d = sys::driver()?;
        let mut ctx = std::ptr::null_mut();
        d.check(
            unsafe { (d.cuDevicePrimaryCtxRetain)(&mut ctx, dev) },
            "contexte",
        )?;
        d.check(unsafe { (d.cuCtxSetCurrent)(ctx) }, "contexte")?;
        let stream = || -> Result<CUstream, BackendError> {
            let mut s = std::ptr::null_mut();
            d.check(
                unsafe { (d.cuStreamCreate)(&mut s, sys::STREAM_NON_BLOCKING) },
                "création d'un stream",
            )?;
            Ok(s)
        };
        let (compute, transfer) = (stream()?, stream()?);
        let attr = |a| {
            let mut v = 0;
            unsafe { (d.cuDeviceGetAttribute)(&mut v, a, dev) };
            v
        };
        let cc = attr(sys::ATTR_CC_MAJOR) * 10 + attr(sys::ATTR_CC_MINOR);
        let pool = Arc::new(BufPool {
            d,
            compute,
            transfer,
            free: Mutex::new((Vec::new(), 0)),
        });
        let dev = Arc::new(CuDevice {
            info,
            d,
            ctx,
            compute,
            transfer,
            cc,
            pool,
            kernels: Mutex::new(HashMap::new()),
            pinned_pool: Mutex::new(HashMap::new()),
        });
        open.insert(index, dev.clone());
        Ok(dev)
    }

    fn bind(&self) -> Result<(), BackendError> {
        self.d
            .check(unsafe { (self.d.cuCtxSetCurrent)(self.ctx) }, "contexte")
    }

    fn record(&self, stream: CUstream) -> Result<CuEvent, BackendError> {
        let mut ev = std::ptr::null_mut();
        self.d.check(
            unsafe { (self.d.cuEventCreate)(&mut ev, sys::EVENT_DISABLE_TIMING) },
            "événement",
        )?;
        let e = CuEvent { d: self.d, ev };
        self.d
            .check(unsafe { (self.d.cuEventRecord)(ev, stream) }, "événement")?;
        Ok(e)
    }

    fn after(&self, stream: CUstream, wait: &[&CuEvent]) -> Result<(), BackendError> {
        for e in wait {
            self.d.check(
                unsafe { (self.d.cuStreamWaitEvent)(stream, e.ev, 0) },
                "attente d'un événement",
            )?;
        }
        Ok(())
    }

    /// Compile (avec cache) le code CUDA C et renvoie le kernel `entry`.
    pub fn compile(&self, code: &str, entry: &str) -> Result<Arc<CuKernel>, BackendError> {
        let mut cache = self.kernels.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(k) = cache.get(code) {
            return Ok(k.clone());
        }
        let ptx = match ptx_cache::get(code, self.cc) {
            Some(p) => p,
            None => {
                let p = self.ptx(code)?;
                ptx_cache::put(code, self.cc, &p);
                p
            }
        };
        self.bind()?;
        let mut module = std::ptr::null_mut();
        self.d.check(
            unsafe { (self.d.cuModuleLoadData)(&mut module, ptx.as_ptr().cast()) },
            "chargement du PTX (driver plus ancien que NVRTC ? mettez le driver NVIDIA à jour)",
        )?;
        let name = CString::new(entry).unwrap();
        let mut func = std::ptr::null_mut();
        if let Err(e) = self.d.check(
            unsafe { (self.d.cuModuleGetFunction)(&mut func, module, name.as_ptr()) },
            "kernel introuvable dans le module",
        ) {
            unsafe { (self.d.cuModuleUnload)(module) };
            return Err(e);
        }
        let k = Arc::new(CuKernel {
            d: self.d,
            ctx: self.ctx,
            module,
            func,
        });
        cache.insert(code.to_string(), k.clone());
        Ok(k)
    }

    /// CUDA C → PTX avec NVRTC, pour l'architecture du GPU (ou la plus récente connue de NVRTC : le
    /// driver compile alors le PTX pour le GPU).
    fn ptx(&self, code: &str) -> Result<Vec<u8>, BackendError> {
        let n = sys::nvrtc()?;
        let archs = n.archs();
        let arch = archs
            .iter()
            .copied()
            .filter(|a| *a <= self.cc)
            .max()
            .unwrap_or(self.cc);
        let src = CString::new(code).map_err(|_| BackendError("code CUDA invalide".into()))?;
        let fname = CString::new("gpusync.cu").unwrap();
        let mut prog = std::ptr::null_mut();
        let r = unsafe {
            (n.nvrtcCreateProgram)(
                &mut prog,
                src.as_ptr(),
                fname.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if r != 0 {
            return Err(BackendError(format!("NVRTC : création ({})", n.error(r))));
        }
        // Pas de --use_fast_math : la précision prime (résultats comparés à NumPy), comme en OpenCL.
        let opts: Vec<CString> = [
            format!("--gpu-architecture=compute_{arch}"),
            "-default-device".into(),
        ]
        .into_iter()
        .map(|o| CString::new(o).unwrap())
        .collect();
        let optp: Vec<*const i8> = opts.iter().map(|o| o.as_ptr()).collect();
        let r = unsafe { (n.nvrtcCompileProgram)(prog, optp.len() as i32, optp.as_ptr()) };
        let result = if r != 0 {
            let mut size = 0usize;
            unsafe { (n.nvrtcGetProgramLogSize)(prog, &mut size) };
            let mut log = vec![0u8; size.max(1)];
            unsafe { (n.nvrtcGetProgramLog)(prog, log.as_mut_ptr().cast()) };
            Err(BackendError(format!(
                "échec de compilation CUDA sur {} ({}) :\n{}",
                self.info.name,
                n.error(r),
                String::from_utf8_lossy(&log).trim_end_matches('\0')
            )))
        } else {
            let mut size = 0usize;
            unsafe { (n.nvrtcGetPTXSize)(prog, &mut size) };
            let mut ptx = vec![0u8; size];
            unsafe { (n.nvrtcGetPTX)(prog, ptx.as_mut_ptr().cast()) };
            Ok(ptx)
        };
        unsafe { (n.nvrtcDestroyProgram)(&mut prog) };
        result
    }

    /// Buffer en mémoire du périphérique.
    pub fn alloc(&self, bytes: usize) -> Result<CuBuffer, BackendError> {
        self.bind()?;
        let cap = capacity(bytes);
        let ptr = match self.pool.take(cap) {
            Some(p) => p,
            None => {
                let mut ptr = 0;
                let mut r = unsafe { (self.d.cuMemAlloc_v2)(&mut ptr, cap) };
                if r != 0 {
                    // Mémoire pleine : rendre la réserve au driver, puis réessayer.
                    self.trim();
                    r = unsafe { (self.d.cuMemAlloc_v2)(&mut ptr, cap) };
                }
                self.d.check(r, "allocation")?;
                ptr
            }
        };
        Ok(CuBuffer {
            d: self.d,
            ctx: self.ctx,
            pool: self.pool.clone(),
            cap,
            ptr,
            bytes,
        })
    }

    /// Libère les buffers en réserve (après la fin des calculs en file).
    fn trim(&self) {
        unsafe { (self.d.cuCtxSynchronize)() };
        let mut g = self.pool.free.lock().unwrap_or_else(|e| e.into_inner());
        for (_, ptr, evs) in g.0.drain(..) {
            unsafe {
                for e in evs {
                    (self.d.cuEventDestroy_v2)(e);
                }
                (self.d.cuMemFree_v2)(ptr);
            }
        }
        g.1 = 0;
    }

    /// Tampon épinglé d'au moins `bytes` octets (pris dans le pool si possible).
    pub fn pinned(&self, bytes: usize) -> Result<Pinned, BackendError> {
        if let Some(p) = self
            .pinned_pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(&bytes)
            .and_then(Vec::pop)
        {
            return Ok(p);
        }
        self.bind()?;
        let mut p: *mut c_void = std::ptr::null_mut();
        self.d.check(
            unsafe { (self.d.cuMemHostAlloc)(&mut p, bytes.max(4), 0) },
            "tampon épinglé",
        )?;
        Ok(Pinned {
            d: self.d,
            ctx: self.ctx,
            ptr: p.cast(),
            bytes,
        })
    }

    /// Rend un tampon épinglé au pool.
    pub fn recycle(&self, p: Pinned) {
        self.pinned_pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(p.bytes)
            .or_default()
            .push(p);
    }

    /// Copie hôte → périphérique (stream de transfert, non bloquante), après `wait`.
    ///
    /// # Safety
    /// `src` doit rester valide jusqu'à la fin de l'événement renvoyé (et être épinglée pour que la
    /// copie soit asynchrone).
    pub unsafe fn write_at(
        &self,
        dst: &CuBuffer,
        offset: usize,
        src: *const u8,
        bytes: usize,
        wait: &[&CuEvent],
    ) -> Result<CuEvent, BackendError> {
        self.bind()?;
        self.after(self.transfer, wait)?;
        self.d.check(
            unsafe {
                (self.d.cuMemcpyHtoDAsync_v2)(
                    dst.ptr + offset as u64,
                    src.cast(),
                    bytes,
                    self.transfer,
                )
            },
            "écriture",
        )?;
        self.record(self.transfer)
    }

    /// Copie périphérique → hôte (stream de transfert, non bloquante), après `wait`.
    ///
    /// # Safety
    /// `dst` doit rester valide jusqu'à la fin de l'événement renvoyé.
    pub unsafe fn read_at(
        &self,
        src: &CuBuffer,
        offset: usize,
        dst: *mut u8,
        bytes: usize,
        wait: &[&CuEvent],
    ) -> Result<CuEvent, BackendError> {
        self.bind()?;
        self.after(self.transfer, wait)?;
        self.d.check(
            unsafe {
                (self.d.cuMemcpyDtoHAsync_v2)(
                    dst.cast(),
                    src.ptr + offset as u64,
                    bytes,
                    self.transfer,
                )
            },
            "lecture",
        )?;
        self.record(self.transfer)
    }

    /// Copie périphérique → périphérique (stream de transfert), après `wait`.
    pub fn copy_at(
        &self,
        src: &CuBuffer,
        src_off: usize,
        dst: &CuBuffer,
        dst_off: usize,
        bytes: usize,
        wait: &[&CuEvent],
    ) -> Result<CuEvent, BackendError> {
        self.bind()?;
        self.after(self.transfer, wait)?;
        self.d.check(
            unsafe {
                (self.d.cuMemcpyDtoDAsync_v2)(
                    dst.ptr + dst_off as u64,
                    src.ptr + src_off as u64,
                    bytes,
                    self.transfer,
                )
            },
            "copie",
        )?;
        self.record(self.transfer)
    }

    /// Met le kernel en file de calcul sur `n` éléments, après `wait`.
    pub fn launch(
        &self,
        k: &CuKernel,
        args: &[CuArg],
        n: usize,
        wait: &[&CuEvent],
    ) -> Result<CuEvent, BackendError> {
        self.bind()?;
        self.after(self.compute, wait)?;
        // Valeurs des arguments (8 octets chacune) et tableau de pointeurs vers elles.
        let mut vals: Vec<[u8; 8]> = args
            .iter()
            .map(|a| {
                let mut v = [0u8; 8];
                match a {
                    CuArg::Buf(b) => v.copy_from_slice(&b.ptr.to_ne_bytes()),
                    CuArg::I32(x) => v[..4].copy_from_slice(&x.to_ne_bytes()),
                    CuArg::U32(x) => v[..4].copy_from_slice(&x.to_ne_bytes()),
                    CuArg::F32(x) => v[..4].copy_from_slice(&x.to_ne_bytes()),
                    CuArg::F64(x) => v.copy_from_slice(&x.to_ne_bytes()),
                    CuArg::I64(x) => v.copy_from_slice(&x.to_ne_bytes()),
                    CuArg::U64(x) => v.copy_from_slice(&x.to_ne_bytes()),
                }
                v
            })
            .collect();
        let mut ptrs: Vec<*mut c_void> = vals.iter_mut().map(|v| v.as_mut_ptr().cast()).collect();
        let grid = (n as u64).div_ceil(BLOCK as u64).max(1);
        if grid > i32::MAX as u64 {
            return Err(BackendError(format!(
                "{n} éléments : trop pour une seule grille CUDA"
            )));
        }
        self.d.check(
            unsafe {
                (self.d.cuLaunchKernel)(
                    k.func,
                    grid as u32,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    self.compute,
                    ptrs.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "lancement du kernel",
        )?;
        self.record(self.compute)
    }

    /// Les commandes CUDA partent sans attendre : rien à faire (pour l'interface commune).
    pub fn flush(&self) -> Result<(), BackendError> {
        Ok(())
    }

    /// Attend la fin de tout ce qui est en file (calcul et transferts).
    pub fn finish(&self) -> Result<(), BackendError> {
        self.bind()?;
        self.d
            .check(unsafe { (self.d.cuCtxSynchronize)() }, "synchronisation")
    }
}

/// Cache disque du PTX produit par NVRTC (~100 ms par kernel et par processus sinon). La suite
/// (PTX → code machine) est déjà mise en cache par le driver. Dossier : `GPUSYNC_CACHE_DIR`, sinon
/// le cache utilisateur (`%LOCALAPPDATA%\gpusync`, `~/.cache/gpusync`, `~/Library/Caches/gpusync`) ;
/// `GPUSYNC_CACHE=0` le désactive. Un fichier illisible est simplement recompilé.
mod ptx_cache {
    use std::path::PathBuf;

    const VERSION: &str = "gpusync-ptx-1";

    fn dir() -> Option<PathBuf> {
        if std::env::var("GPUSYNC_CACHE").is_ok_and(|v| v == "0") {
            return None;
        }
        if let Some(d) = std::env::var_os("GPUSYNC_CACHE_DIR") {
            return Some(PathBuf::from(d).join("cuda"));
        }
        let base = if cfg!(windows) {
            std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
        } else if cfg!(target_os = "macos") {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Caches"))
        } else {
            std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        }?;
        Some(base.join("gpusync").join("cuda"))
    }

    /// FNV-1a 64 bits : clé stable entre exécutions et versions de Rust.
    fn key(code: &str, cc: i32) -> String {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in VERSION.bytes().chain(cc.to_le_bytes()).chain(code.bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}.ptx")
    }

    pub fn get(code: &str, cc: i32) -> Option<Vec<u8>> {
        let p = std::fs::read(dir()?.join(key(code, cc))).ok()?;
        // PTX terminé par un zéro (attendu par cuModuleLoadData).
        (p.last() == Some(&0) && p.starts_with(b"//")).then_some(p)
    }

    pub fn put(code: &str, cc: i32, ptx: &[u8]) {
        let Some(d) = dir() else { return };
        if std::fs::create_dir_all(&d).is_err() {
            return;
        }
        let tmp = d.join(format!("{}.{}.tmp", key(code, cc), std::process::id()));
        if std::fs::write(&tmp, ptx).is_ok()
            && std::fs::rename(&tmp, d.join(key(code, cc))).is_err()
        {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}
