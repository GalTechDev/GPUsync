//! Interface commune des backends à files et événements (OpenCL, CUDA) : le moteur de transferts
//! (modes Résident et par blocs, §9) est écrit une fois, pour les deux.

use std::sync::Arc;

use gpusync_backend_cuda::{CuArg, CuBuffer, CuDevice, CuEvent, CuKernel};
use gpusync_backend_opencl::{ClArg, ClBuffer, ClDevice, ClEvent, ClKernel};
use gpusync_runtime::{BackendError, DeviceInfo};

use crate::exec::Storage;

/// Argument de lancement, dans l'ordre des paramètres du kernel.
pub(crate) enum GArg<'a, B> {
    Buf(&'a B),
    I32(i32),
    U32(u32),
    F32(f32),
    F64(f64),
    I64(i64),
    U64(u64),
}

type Res<T> = Result<T, BackendError>;

/// Périphérique avec une file de calcul et une file de transferts, chaînées par événements.
pub(crate) trait Gpu {
    type Buf;
    type Ev;
    type Kern;
    type Pin;

    fn info(&self) -> &DeviceInfo;
    fn alloc(&self, bytes: usize) -> Res<Self::Buf>;
    fn pinned(&self, bytes: usize) -> Res<Self::Pin>;
    fn pin_ptr(p: &Self::Pin) -> *mut u8;
    fn recycle(&self, p: Self::Pin);
    /// # Safety
    /// `src` doit rester valide jusqu'à la fin de l'événement renvoyé.
    unsafe fn write_at(
        &self,
        dst: &Self::Buf,
        off: usize,
        src: *const u8,
        bytes: usize,
        wait: &[&Self::Ev],
    ) -> Res<Self::Ev>;
    /// # Safety
    /// `dst` doit rester valide jusqu'à la fin de l'événement renvoyé.
    unsafe fn read_at(
        &self,
        src: &Self::Buf,
        off: usize,
        dst: *mut u8,
        bytes: usize,
        wait: &[&Self::Ev],
    ) -> Res<Self::Ev>;
    fn copy_at(
        &self,
        src: &Self::Buf,
        src_off: usize,
        dst: &Self::Buf,
        dst_off: usize,
        bytes: usize,
        wait: &[&Self::Ev],
    ) -> Res<Self::Ev>;
    fn launch(
        &self,
        k: &Self::Kern,
        args: &[GArg<'_, Self::Buf>],
        n: usize,
        wait: &[&Self::Ev],
    ) -> Res<Self::Ev>;
    fn flush(&self) -> Res<()>;
    fn wait(e: &Self::Ev) -> Res<()>;
    fn finish(&self) -> Res<()>;
    /// Résultat resté sur le périphérique.
    fn storage(buf: Self::Buf, done: Arc<Self::Ev>) -> Storage;
    /// Buffer et événement de fin d'un résultat resté sur ce type de périphérique.
    fn resident(s: &Storage) -> Option<(&Self::Buf, &Self::Ev)>;
}

impl Gpu for ClDevice {
    type Buf = ClBuffer;
    type Ev = ClEvent;
    type Kern = ClKernel;
    type Pin = gpusync_backend_opencl::Pinned;

    fn info(&self) -> &DeviceInfo {
        &self.info
    }
    fn alloc(&self, bytes: usize) -> Res<ClBuffer> {
        ClDevice::alloc(self, bytes)
    }
    fn pinned(&self, bytes: usize) -> Res<Self::Pin> {
        ClDevice::pinned(self, bytes)
    }
    fn pin_ptr(p: &Self::Pin) -> *mut u8 {
        p.ptr
    }
    fn recycle(&self, p: Self::Pin) {
        ClDevice::recycle(self, p)
    }
    unsafe fn write_at(
        &self,
        dst: &ClBuffer,
        off: usize,
        src: *const u8,
        bytes: usize,
        wait: &[&ClEvent],
    ) -> Res<ClEvent> {
        unsafe { ClDevice::write_at(self, dst, off, src, bytes, wait) }
    }
    unsafe fn read_at(
        &self,
        src: &ClBuffer,
        off: usize,
        dst: *mut u8,
        bytes: usize,
        wait: &[&ClEvent],
    ) -> Res<ClEvent> {
        unsafe { ClDevice::read_at(self, src, off, dst, bytes, wait) }
    }
    fn copy_at(
        &self,
        src: &ClBuffer,
        src_off: usize,
        dst: &ClBuffer,
        dst_off: usize,
        bytes: usize,
        wait: &[&ClEvent],
    ) -> Res<ClEvent> {
        ClDevice::copy_at(self, src, src_off, dst, dst_off, bytes, wait)
    }
    fn launch(
        &self,
        k: &ClKernel,
        args: &[GArg<'_, ClBuffer>],
        n: usize,
        wait: &[&ClEvent],
    ) -> Res<ClEvent> {
        let a: Vec<ClArg> = args
            .iter()
            .map(|a| match a {
                GArg::Buf(b) => ClArg::Buf(b),
                GArg::I32(v) => ClArg::I32(*v),
                GArg::U32(v) => ClArg::U32(*v),
                GArg::F32(v) => ClArg::F32(*v),
                GArg::F64(v) => ClArg::F64(*v),
                GArg::I64(v) => ClArg::I64(*v),
                GArg::U64(v) => ClArg::U64(*v),
            })
            .collect();
        ClDevice::launch(self, k, &a, n, wait)
    }
    fn flush(&self) -> Res<()> {
        ClDevice::flush(self)
    }
    fn wait(e: &ClEvent) -> Res<()> {
        gpusync_backend_opencl::wait(e)
    }
    fn finish(&self) -> Res<()> {
        ClDevice::finish(self)
    }
    fn storage(buf: ClBuffer, done: Arc<ClEvent>) -> Storage {
        Storage::Cl { buf, done }
    }
    fn resident(s: &Storage) -> Option<(&ClBuffer, &ClEvent)> {
        match s {
            Storage::Cl { buf, done } => Some((buf, done)),
            // Résultat Direct d'un seul bloc (vérifié par l'appelant) : son buffer resert tel quel.
            Storage::ClDirect {
                bufs,
                done: Some(done),
                ..
            } => bufs.first().map(|b| (b, &**done)),
            _ => None,
        }
    }
}

impl Gpu for CuDevice {
    type Buf = CuBuffer;
    type Ev = CuEvent;
    type Kern = CuKernel;
    type Pin = gpusync_backend_cuda::Pinned;

    fn info(&self) -> &DeviceInfo {
        &self.info
    }
    fn alloc(&self, bytes: usize) -> Res<CuBuffer> {
        CuDevice::alloc(self, bytes)
    }
    fn pinned(&self, bytes: usize) -> Res<Self::Pin> {
        CuDevice::pinned(self, bytes)
    }
    fn pin_ptr(p: &Self::Pin) -> *mut u8 {
        p.ptr
    }
    fn recycle(&self, p: Self::Pin) {
        CuDevice::recycle(self, p)
    }
    unsafe fn write_at(
        &self,
        dst: &CuBuffer,
        off: usize,
        src: *const u8,
        bytes: usize,
        wait: &[&CuEvent],
    ) -> Res<CuEvent> {
        unsafe { CuDevice::write_at(self, dst, off, src, bytes, wait) }
    }
    unsafe fn read_at(
        &self,
        src: &CuBuffer,
        off: usize,
        dst: *mut u8,
        bytes: usize,
        wait: &[&CuEvent],
    ) -> Res<CuEvent> {
        unsafe { CuDevice::read_at(self, src, off, dst, bytes, wait) }
    }
    fn copy_at(
        &self,
        src: &CuBuffer,
        src_off: usize,
        dst: &CuBuffer,
        dst_off: usize,
        bytes: usize,
        wait: &[&CuEvent],
    ) -> Res<CuEvent> {
        CuDevice::copy_at(self, src, src_off, dst, dst_off, bytes, wait)
    }
    fn launch(
        &self,
        k: &CuKernel,
        args: &[GArg<'_, CuBuffer>],
        n: usize,
        wait: &[&CuEvent],
    ) -> Res<CuEvent> {
        let a: Vec<CuArg> = args
            .iter()
            .map(|a| match a {
                GArg::Buf(b) => CuArg::Buf(b),
                GArg::I32(v) => CuArg::I32(*v),
                GArg::U32(v) => CuArg::U32(*v),
                GArg::F32(v) => CuArg::F32(*v),
                GArg::F64(v) => CuArg::F64(*v),
                GArg::I64(v) => CuArg::I64(*v),
                GArg::U64(v) => CuArg::U64(*v),
            })
            .collect();
        CuDevice::launch(self, k, &a, n, wait)
    }
    fn flush(&self) -> Res<()> {
        CuDevice::flush(self)
    }
    fn wait(e: &CuEvent) -> Res<()> {
        e.wait()
    }
    fn finish(&self) -> Res<()> {
        CuDevice::finish(self)
    }
    fn storage(buf: CuBuffer, done: Arc<CuEvent>) -> Storage {
        Storage::Cu { buf, done }
    }
    fn resident(s: &Storage) -> Option<(&CuBuffer, &CuEvent)> {
        match s {
            Storage::Cu { buf, done } => Some((buf, done)),
            _ => None,
        }
    }
}
