//! Minimal bindings, loaded dynamically, to the CUDA driver (`nvcuda.dll`, `libcuda.so.1`) and
//! NVRTC. No CUDA library is needed at build time; when they are missing at run time, the
//! backend is simply unavailable, with the reason (never a panic).

#![allow(non_snake_case)] // names of the C API functions

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use gpusync_runtime::BackendError;
use libloading::Library;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUdeviceptr = u64;
pub type NvrtcProgram = *mut c_void;

pub const ATTR_MAX_THREADS_PER_BLOCK: c_int = 1;
pub const ATTR_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
pub const ATTR_INTEGRATED: c_int = 18;
pub const ATTR_CC_MAJOR: c_int = 75;
pub const ATTR_CC_MINOR: c_int = 76;
pub const STREAM_NON_BLOCKING: c_uint = 1;
pub const EVENT_DISABLE_TIMING: c_uint = 2;

macro_rules! api {
    ($name:ident { $($f:ident : fn($($a:ty),*) -> $r:ty),* $(,)? }) => {
        #[allow(non_snake_case)]
        pub struct $name {
            _lib: Library,
            $(pub $f: unsafe extern "C" fn($($a),*) -> $r,)*
        }
        impl $name {
            unsafe fn load(lib: Library) -> Result<Self, String> {
                $(
                    let $f = unsafe {
                        *lib.get::<unsafe extern "C" fn($($a),*) -> $r>(concat!(stringify!($f), "\0").as_bytes())
                            .map_err(|e| format!("{}: {e}", stringify!($f)))?
                    };
                )*
                Ok(Self { _lib: lib, $($f,)* })
            }
        }
    };
}

api!(Driver {
    cuInit: fn(c_uint) -> CUresult,
    cuDriverGetVersion: fn(*mut c_int) -> CUresult,
    cuGetErrorString: fn(CUresult, *mut *const c_char) -> CUresult,
    cuDeviceGetCount: fn(*mut c_int) -> CUresult,
    cuDeviceGet: fn(*mut CUdevice, c_int) -> CUresult,
    cuDeviceGetName: fn(*mut c_char, c_int, CUdevice) -> CUresult,
    cuDeviceTotalMem_v2: fn(*mut usize, CUdevice) -> CUresult,
    cuDeviceGetAttribute: fn(*mut c_int, c_int, CUdevice) -> CUresult,
    cuDevicePrimaryCtxRetain: fn(*mut CUcontext, CUdevice) -> CUresult,
    cuCtxSetCurrent: fn(CUcontext) -> CUresult,
    cuCtxSynchronize: fn() -> CUresult,
    cuStreamCreate: fn(*mut CUstream, c_uint) -> CUresult,
    cuStreamWaitEvent: fn(CUstream, CUevent, c_uint) -> CUresult,
    cuMemAlloc_v2: fn(*mut CUdeviceptr, usize) -> CUresult,
    cuMemFree_v2: fn(CUdeviceptr) -> CUresult,
    cuMemHostAlloc: fn(*mut *mut c_void, usize, c_uint) -> CUresult,
    cuMemFreeHost: fn(*mut c_void) -> CUresult,
    cuMemcpyHtoDAsync_v2: fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult,
    cuMemcpyDtoHAsync_v2: fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult,
    cuMemcpyDtoDAsync_v2: fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult,
    cuEventCreate: fn(*mut CUevent, c_uint) -> CUresult,
    cuEventRecord: fn(CUevent, CUstream) -> CUresult,
    cuEventSynchronize: fn(CUevent) -> CUresult,
    cuEventQuery: fn(CUevent) -> CUresult,
    cuEventDestroy_v2: fn(CUevent) -> CUresult,
    cuModuleLoadData: fn(*mut CUmodule, *const c_void) -> CUresult,
    cuModuleGetFunction: fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    cuModuleUnload: fn(CUmodule) -> CUresult,
    cuLaunchKernel: fn(CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, *mut *mut c_void, *mut *mut c_void) -> CUresult,
});

api!(Nvrtc {
    nvrtcVersion: fn(*mut c_int, *mut c_int) -> c_int,
    nvrtcGetErrorString: fn(c_int) -> *const c_char,
    nvrtcCreateProgram: fn(*mut NvrtcProgram, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> c_int,
    nvrtcDestroyProgram: fn(*mut NvrtcProgram) -> c_int,
    nvrtcCompileProgram: fn(NvrtcProgram, c_int, *const *const c_char) -> c_int,
    nvrtcGetProgramLogSize: fn(NvrtcProgram, *mut usize) -> c_int,
    nvrtcGetProgramLog: fn(NvrtcProgram, *mut c_char) -> c_int,
    nvrtcGetPTXSize: fn(NvrtcProgram, *mut usize) -> c_int,
    nvrtcGetPTX: fn(NvrtcProgram, *mut c_char) -> c_int,
    nvrtcGetNumSupportedArchs: fn(*mut c_int) -> c_int,
    nvrtcGetSupportedArchs: fn(*mut c_int) -> c_int,
});

// The function pointers and the library can be used from every thread.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}
unsafe impl Send for Nvrtc {}
unsafe impl Sync for Nvrtc {}

impl Driver {
    pub fn check(&self, r: CUresult, what: &str) -> Result<(), BackendError> {
        if r == 0 {
            return Ok(());
        }
        let mut s: *const c_char = std::ptr::null();
        let msg = unsafe {
            if (self.cuGetErrorString)(r, &mut s) == 0 && !s.is_null() {
                CStr::from_ptr(s).to_string_lossy().into_owned()
            } else {
                format!("code {r}")
            }
        };
        Err(BackendError(format!("CUDA: {what} ({msg})")))
    }
}

/// CUDA driver, loaded and initialized once.
pub fn driver() -> Result<&'static Driver, BackendError> {
    static D: OnceLock<Result<Driver, String>> = OnceLock::new();
    D.get_or_init(|| {
        let names: &[&str] = if cfg!(windows) {
            &["nvcuda.dll"]
        } else if cfg!(target_os = "macos") {
            return Err("no CUDA driver on macOS (no NVIDIA GPU)".into());
        } else {
            &["libcuda.so.1", "libcuda.so"]
        };
        let lib = names
            .iter()
            .find_map(|n| unsafe { Library::new(n) }.ok())
            .ok_or_else(|| "NVIDIA driver not found (nvcuda / libcuda)".to_string())?;
        let d =
            unsafe { Driver::load(lib) }.map_err(|e| format!("CUDA driver too old: {e}"))?;
        let r = unsafe { (d.cuInit)(0) };
        if r != 0 {
            return Err(d.check(r, "initialization").unwrap_err().0);
        }
        Ok(d)
    })
    .as_ref()
    .map_err(|e| BackendError(e.clone()))
}

/// Folders where NVRTC is looked for: `GPUSYNC_NVRTC` (file), then `GPUSYNC_NVRTC_DIRS` (filled in by
/// the Python package: pip wheels `nvidia-cuda-nvrtc-cu12` / `-cu13`), then the CUDA toolkit
/// (`CUDA_PATH`, `CUDA_PATH_V*`, /usr/local/cuda) and the PATH.
fn nvrtc_candidates() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(v) = std::env::var("GPUSYNC_NVRTC_DIRS") {
        dirs.extend(std::env::split_paths(&v));
    }
    let mut toolkit: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| k == "CUDA_PATH" || k.starts_with("CUDA_PATH_V") || k == "CUDA_HOME")
        .collect();
    toolkit.sort();
    for (_, v) in toolkit.iter().rev() {
        let base = PathBuf::from(v);
        dirs.push(base.join("bin"));
        dirs.push(base.join("lib64"));
    }
    if !cfg!(windows) {
        dirs.push(PathBuf::from("/usr/local/cuda/lib64"));
        dirs.push(PathBuf::from("/usr/lib/x86_64-linux-gnu"));
        dirs.push(PathBuf::from("/usr/lib/aarch64-linux-gnu"));
    }
    if let Ok(p) = std::env::var("PATH") {
        dirs.extend(std::env::split_paths(&p));
    }
    let mut out = Vec::new();
    if let Ok(f) = std::env::var("GPUSYNC_NVRTC") {
        out.push(PathBuf::from(f));
    }
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        let mut found: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if cfg!(windows) {
                    n.starts_with("nvrtc64_") && n.ends_with(".dll") && !n.contains(".alt.")
                } else {
                    n.starts_with("libnvrtc.so") && !n.contains("builtins")
                }
            })
            .collect();
        found.sort();
        out.extend(found.into_iter().rev()); // most recent version first
    }
    out
}

/// Message when NVRTC is missing (driver present).
pub const NVRTC_MISSING: &str = "NVRTC not found: the NVIDIA driver is present, but the library that compiles CUDA kernels \
     is missing. Install it with `pip install nvidia-cuda-nvrtc-cu12` (or the CUDA toolkit)";

/// Is NVRTC present (only looks for the files, without loading it)?
pub fn nvrtc_present() -> bool {
    !nvrtc_candidates().is_empty()
}

/// NVRTC (CUDA C → PTX compiler, a library) loaded once.
pub fn nvrtc() -> Result<&'static Nvrtc, BackendError> {
    static N: OnceLock<Result<Nvrtc, String>> = OnceLock::new();
    N.get_or_init(|| {
        let mut tried = Vec::new();
        for path in nvrtc_candidates() {
            if let Some(lib) = unsafe { open_beside(&path) } {
                match unsafe { Nvrtc::load(lib) } {
                    Ok(n) => return Ok(n),
                    Err(e) => tried.push(format!("{} ({e})", path.display())),
                }
            }
        }
        Err(format!(
            "{NVRTC_MISSING}{}",
            if tried.is_empty() {
                String::new()
            } else {
                format!("; tried: {}", tried.join(", "))
            }
        ))
    })
    .as_ref()
    .map_err(|e| BackendError(e.clone()))
}

/// Loads a library so that its dependencies (e.g. `nvrtc-builtins`) can be found in its
/// own folder.
unsafe fn open_beside(path: &Path) -> Option<Library> {
    #[cfg(windows)]
    {
        use libloading::os::windows::{
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, Library as WinLib,
        };
        let flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
        let lib = unsafe { WinLib::load_with_flags(path, flags) }.ok()?;
        // NVRTC loads nvrtc-builtins on demand, by name: its folder must be searched.
        if let Some(dir) = path.parent() {
            let p = std::env::var_os("PATH").unwrap_or_default();
            if !std::env::split_paths(&p).any(|x| x == dir) {
                let mut all = vec![dir.to_path_buf()];
                all.extend(std::env::split_paths(&p));
                if let Ok(joined) = std::env::join_paths(all) {
                    unsafe { std::env::set_var("PATH", joined) };
                }
            }
        }
        Some(lib.into())
    }
    #[cfg(not(windows))]
    {
        unsafe { Library::new(path) }.ok()
    }
}

impl Nvrtc {
    pub fn error(&self, r: c_int) -> String {
        unsafe {
            let s = (self.nvrtcGetErrorString)(r);
            if s.is_null() {
                format!("code {r}")
            } else {
                CStr::from_ptr(s).to_string_lossy().into_owned()
            }
        }
    }

    pub fn version(&self) -> (i32, i32) {
        let (mut a, mut b) = (0, 0);
        unsafe { (self.nvrtcVersion)(&mut a, &mut b) };
        (a, b)
    }

    /// Virtual architectures accepted by this NVRTC (e.g. 89 for compute_89).
    pub fn archs(&self) -> Vec<i32> {
        let mut n = 0;
        unsafe {
            if (self.nvrtcGetNumSupportedArchs)(&mut n) != 0 || n <= 0 {
                return Vec::new();
            }
            let mut v = vec![0; n as usize];
            if (self.nvrtcGetSupportedArchs)(v.as_mut_ptr()) != 0 {
                return Vec::new();
            }
            v
        }
    }
}
