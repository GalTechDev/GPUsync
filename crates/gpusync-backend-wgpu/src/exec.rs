//! Exécution wgpu : un `Device` par adaptateur, pipelines en cache, buffers de stockage.
//!
//! Disposition des bindings (voir gpusync-codegen) : tableaux d'entrée, sortie, puis `gs_params`
//! (scalaires et `n`, un mot de 32 bits chacun). La grille est 2D pour dépasser 65 535 groupes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use gpusync_runtime::{BackendError, DeviceInfo};

pub const WORKGROUP_SIZE: u32 = 256;

pub struct WgDevice {
    pub info: DeviceInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    max_groups_per_dim: u32,
    kernels: Mutex<HashMap<String, Arc<WgKernel>>>,
    staging_pool: Mutex<HashMap<(usize, bool), Vec<Staging>>>,
    /// Clé dans `OPEN` (ex. `vulkan:1`).
    key: String,
    /// Raison de la perte du périphérique par le driver (TDR Windows, plantage…), si perdu.
    lost: Arc<Mutex<Option<String>>>,
}

pub struct WgKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    n_inputs: usize,
    n_outputs: usize,
    has_fparams: bool,
}

pub struct WgBuffer {
    buf: wgpu::Buffer,
    pub bytes: usize,
}

static OPEN: OnceLock<Mutex<HashMap<String, Arc<WgDevice>>>> = OnceLock::new();

fn err(what: &str, e: impl std::fmt::Display) -> BackendError {
    BackendError(format!("wgpu : {what} ({e})"))
}

/// Taille alignée sur 4 octets (contrainte des copies wgpu), jamais nulle.
fn aligned(bytes: usize) -> u64 {
    (bytes.max(4) as u64).div_ceil(4) * 4
}

impl WgDevice {
    /// Erreur claire si le driver a perdu le périphérique ; il est alors oublié et sera rouvert
    /// au prochain `open`.
    fn check(&self) -> Result<(), BackendError> {
        let lost = self.lost.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let Some(reason) = lost else { return Ok(()) };
        if let Some(open) = OPEN.get() {
            let mut open = open.lock().unwrap_or_else(|e| e.into_inner());
            if open
                .get(&self.key)
                .is_some_and(|d| std::ptr::eq(&**d, self))
            {
                open.remove(&self.key);
            }
        }
        Err(BackendError(format!(
            "wgpu : le driver a perdu le périphérique {} ({reason}). Sous Windows, un calcul de plus \
             de ~2 s sur un GPU est interrompu (TDR) : réduisez la taille de l'appel ou choisissez un GPU \
             plus rapide. Le périphérique sera rouvert au prochain appel.",
            self.info.name
        )))
    }

    /// Buffer de stockage initialisé (sans `mapped_at_creation`, qui panique si le périphérique
    /// est perdu).
    fn storage_init(&self, contents: &[u8]) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: aligned(contents.len()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut data = contents.to_vec();
        data.resize(aligned(contents.len()) as usize, 0);
        self.queue.write_buffer(&buf, 0, &data);
        buf
    }

    /// Ouvre (une fois par processus) le périphérique `<api>:<index>` (ex. `metal:0`, `vulkan:1`).
    pub fn open(api: &str, index: usize) -> Result<Arc<WgDevice>, BackendError> {
        let key = format!("{api}:{index}");
        let mut open = OPEN
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(d) = open.get(&key) {
            return Ok(d.clone());
        }
        let st = crate::state()?;
        let (adapter_idx, info) = st
            .devices
            .iter()
            .find(|(_, d)| d.backend == api && d.index == index)
            .ok_or_else(|| BackendError(format!("périphérique {key} introuvable")))?;
        let adapter = &st.adapters[*adapter_idx];
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("gpusync"),
            // f64 en WGSL (Vulkan) : extension native de wgpu, demandée seulement si disponible.
            required_features: if info.f64 {
                wgpu::Features::SHADER_F64
            } else {
                wgpu::Features::empty()
            },
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| err("ouverture du périphérique", e))?;
        // Par défaut wgpu panique sur une erreur non capturée : on la journalise plutôt.
        device.on_uncaptured_error(Arc::new(|e| {
            eprintln!("[gpusync] erreur wgpu non capturée : {e}")
        }));
        let lost: Arc<Mutex<Option<String>>> = Arc::default();
        let flag = lost.clone();
        device.set_device_lost_callback(move |reason, msg| {
            *flag.lock().unwrap_or_else(|e| e.into_inner()) = Some(format!("{reason:?} : {msg}"));
        });
        let d = Arc::new(WgDevice {
            info: info.clone(),
            device,
            queue,
            max_groups_per_dim: limits.max_compute_workgroups_per_dimension.max(1),
            kernels: Mutex::new(HashMap::new()),
            staging_pool: Mutex::new(HashMap::new()),
            key: key.clone(),
            lost,
        });
        open.insert(key, d.clone());
        Ok(d)
    }

    /// Compile (avec cache) un module WGSL dont le point d'entrée est `main`.
    pub fn compile(
        &self,
        code: &str,
        n_inputs: usize,
        n_outputs: usize,
        has_fparams: bool,
    ) -> Result<Arc<WgKernel>, BackendError> {
        let mut cache = self.kernels.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(k) = cache.get(code) {
            return Ok(k.clone());
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        // Disposition explicite : un binding déclaré mais inutilisé reste valide.
        let mut entries: Vec<_> = (0..n_inputs as u32).map(|b| storage(b, true)).collect();
        let first_param = (n_inputs + n_outputs) as u32;
        entries.extend((n_inputs as u32..first_param).map(|b| storage(b, false)));
        entries.push(storage(first_param, true));
        if has_fparams {
            entries.push(storage(first_param + 1, true));
        }
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &entries,
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpusync"),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("gpusync"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(BackendError(format!(
                "échec de compilation WGSL sur {} :\n{e}",
                self.info.name
            )));
        }
        let k = Arc::new(WgKernel {
            pipeline,
            layout,
            n_inputs,
            n_outputs,
            has_fparams,
        });
        cache.insert(code.to_string(), k.clone());
        Ok(k)
    }

    pub fn alloc(&self, bytes: usize) -> Result<WgBuffer, BackendError> {
        self.check()?;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: aligned(bytes),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(WgBuffer { buf, bytes })
    }

    /// Copie hôte → périphérique (les données sont copiées immédiatement par wgpu).
    pub fn write(&self, buf: &WgBuffer, data: &[u8]) -> Result<(), BackendError> {
        if data.len().is_multiple_of(4) {
            self.queue.write_buffer(&buf.buf, 0, data);
        } else {
            let mut padded = data.to_vec();
            padded.resize(data.len().div_ceil(4) * 4, 0);
            self.queue.write_buffer(&buf.buf, 0, &padded);
        }
        Ok(())
    }

    /// Met le kernel en file sur `n` éléments.
    pub fn launch(
        &self,
        k: &WgKernel,
        inputs: &[&WgBuffer],
        outs: &[&WgBuffer],
        params: &[u32],
        fparams: &[f64],
        n: usize,
    ) -> Result<(), BackendError> {
        self.check()?;
        if inputs.len() != k.n_inputs || outs.len() != k.n_outputs {
            return Err(BackendError(
                "nombre de tableaux d'entrée incohérent".into(),
            ));
        }
        // Toute erreur de validation ou de mémoire devient une erreur Python (jamais un résultat
        // faux silencieux, objectif O2).
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let submitted = self.submit(k, inputs, outs, params, fparams, n);
        let errors = [
            pollster::block_on(validation.pop()),
            pollster::block_on(oom.pop()),
        ];
        submitted?;
        if let Some(e) = errors.into_iter().flatten().next() {
            self.check()?;
            return Err(BackendError(format!(
                "wgpu : lancement refusé sur {} ({e})",
                self.info.name
            )));
        }
        Ok(())
    }

    fn submit(
        &self,
        k: &WgKernel,
        inputs: &[&WgBuffer],
        outs: &[&WgBuffer],
        params: &[u32],
        fparams: &[f64],
        n: usize,
    ) -> Result<(), BackendError> {
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.encode(&mut enc, k, inputs, outs, params, fparams, n)?;
        self.queue.submit([enc.finish()]);
        Ok(())
    }

    /// Enregistre le calcul de `k` sur `n` éléments dans `enc`.
    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        enc: &mut wgpu::CommandEncoder,
        k: &WgKernel,
        inputs: &[&WgBuffer],
        outs: &[&WgBuffer],
        params: &[u32],
        fparams: &[f64],
        n: usize,
    ) -> Result<(), BackendError> {
        let params_buf = self.storage_init(bytemuck_u32(params));
        let mut entries: Vec<wgpu::BindGroupEntry> = inputs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.buf.as_entire_binding(),
            })
            .collect();
        for (o, b) in outs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (inputs.len() + o) as u32,
                resource: b.buf.as_entire_binding(),
            });
        }
        let first_param = (inputs.len() + outs.len()) as u32;
        entries.push(wgpu::BindGroupEntry {
            binding: first_param,
            resource: params_buf.as_entire_binding(),
        });
        let fparams_buf = k.has_fparams.then(|| {
            self.storage_init(unsafe {
                std::slice::from_raw_parts(fparams.as_ptr().cast(), std::mem::size_of_val(fparams))
            })
        });
        if let Some(b) = &fparams_buf {
            entries.push(wgpu::BindGroupEntry {
                binding: first_param + 1,
                resource: b.as_entire_binding(),
            });
        }
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &k.layout,
            entries: &entries,
        });

        let groups = (n as u64).div_ceil(WORKGROUP_SIZE as u64);
        let gx = groups.min(self.max_groups_per_dim as u64) as u32;
        let gy = groups.div_ceil(gx.max(1) as u64);
        if gy > self.max_groups_per_dim as u64 {
            return Err(BackendError(format!(
                "{n} éléments : trop pour une seule grille sur {}",
                self.info.name
            )));
        }
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(gx, gy as u32, 1);
        }
        Ok(())
    }

    /// Copie bloquante périphérique → hôte (attend la fin des calculs soumis).
    pub fn read(&self, buf: &WgBuffer, out: &mut [u8]) -> Result<(), BackendError> {
        self.check()?;
        let size = aligned(buf.bytes);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&buf.buf, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let result: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> = Arc::default();
        let r = result.clone();
        staging.map_async(wgpu::MapMode::Read, .., move |res| {
            *r.lock().unwrap() = Some(res)
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| err("attente du GPU", e))?;
        match result.lock().unwrap().take() {
            Some(Ok(())) => {}
            Some(Err(e)) => return Err(err("lecture du résultat", e)),
            None => return Err(BackendError("wgpu : lecture non terminée".into())),
        }
        {
            let view = staging
                .get_mapped_range(..)
                .map_err(|e| err("accès au résultat", e))?;
            out.copy_from_slice(&view[..out.len()]);
        }
        staging.unmap();
        Ok(())
    }

    /// Buffer Metal qui utilise directement la mémoire de l'hôte (zéro-copie, `newBufferWithBytesNoCopy`).
    ///
    /// Renvoie `Ok(None)` quand ce n'est pas possible (autre API que Metal, adresse non alignée sur une
    /// page de 16 Kio, mémoire refusée par Metal) : l'appelant passe alors par les transferts.
    ///
    /// # Safety
    /// `ptr..ptr + bytes` arrondi à la page suivante doit rester valide tant que le buffer existe.
    pub unsafe fn wrap_host(
        &self,
        ptr: *mut u8,
        bytes: usize,
    ) -> Result<Option<WgBuffer>, BackendError> {
        #[cfg(target_vendor = "apple")]
        {
            use objc2_metal::{MTLDevice, MTLResourceOptions};
            const PAGE: usize = 16384;
            if self.info.backend != "metal" || !(ptr as usize).is_multiple_of(PAGE) || bytes == 0 {
                return Ok(None);
            }
            let len = bytes.div_ceil(PAGE) * PAGE; // la dernière page entière reste dans la même page mappée
            let Some(hal_dev) = (unsafe { self.device.as_hal::<wgpu::hal::api::Metal>() }) else {
                return Ok(None);
            };
            let Some(raw) = (unsafe {
                hal_dev
                    .raw_device()
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        std::ptr::NonNull::new_unchecked(ptr.cast()),
                        len,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    )
            }) else {
                return Ok(None);
            };
            let hal_buf = unsafe { wgpu::hal::metal::Device::buffer_from_raw(raw, len as u64) };
            let buf = unsafe {
                self.device.create_buffer_from_hal::<wgpu::hal::api::Metal>(
                    hal_buf,
                    &wgpu::BufferDescriptor {
                        label: Some("gpusync-nocopy"),
                        size: len as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                )
            };
            Ok(Some(WgBuffer { buf, bytes }))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = (ptr, bytes);
            Ok(None)
        }
    }

    // ───────────────────── transferts par blocs (ARCHITECTURE.md §9) ─────────────────────

    fn staging(&self, bytes: usize, upload: bool) -> Staging {
        if let Some(st) = self
            .staging_pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(&(bytes, upload))
            .and_then(Vec::pop)
        {
            return st;
        }
        let usage = if upload {
            wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC
        } else {
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
        };
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpusync-staging"),
            size: aligned(bytes),
            usage,
            mapped_at_creation: upload, // un tampon d'envoi naît prêt à être rempli
        });
        Staging {
            buf,
            bytes,
            mapped: Arc::new(AtomicBool::new(upload)),
            submission: None,
        }
    }

    fn recycle(&self, st: Staging, upload: bool) {
        self.staging_pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry((st.bytes, upload))
            .or_default()
            .push(st);
    }

    /// Attend que `st` soit mappé (sa dernière copie terminée), sans attendre les soumissions suivantes.
    fn wait_mapped(&self, st: &Staging) -> Result<(), BackendError> {
        if st.mapped.load(Ordering::Acquire) {
            return Ok(());
        }
        let ty = match &st.submission {
            Some(idx) => wgpu::PollType::Wait {
                submission_index: Some(idx.clone()),
                timeout: None,
            },
            None => wgpu::PollType::wait_indefinitely(),
        };
        self.device
            .poll(ty)
            .map_err(|e| err("attente d'un transfert", e))?;
        while !st.mapped.load(Ordering::Acquire) {
            self.device
                .poll(wgpu::PollType::Poll)
                .map_err(|e| err("attente d'un transfert", e))?;
            std::thread::yield_now();
        }
        Ok(())
    }

    fn remap(&self, st: &mut Staging, mode: wgpu::MapMode, idx: &wgpu::SubmissionIndex) {
        st.mapped.store(false, Ordering::Release);
        st.submission = Some(idx.clone());
        let flag = st.mapped.clone();
        st.buf.map_async(mode, .., move |r| {
            if r.is_ok() {
                flag.store(true, Ordering::Release);
            }
        });
    }

    /// Envoie `bytes` octets de l'hôte vers `dst` (à partir de `dst_off`) par blocs de `chunk` octets,
    /// à travers un anneau de 3 tampons : la mémoire hôte en plus reste bornée à 3 blocs.
    ///
    /// # Safety
    /// `src` doit pointer vers `bytes` octets lisibles.
    pub unsafe fn upload(
        &self,
        dst: &WgBuffer,
        dst_off: usize,
        src: *const u8,
        bytes: usize,
        chunk: usize,
    ) -> Result<(), BackendError> {
        self.check()?;
        let chunk = chunk.min(bytes).max(4);
        let mut ring: Vec<Staging> = (0..3).map(|_| self.staging(chunk, true)).collect();
        let mut off = 0;
        let mut i = 0;
        while off < bytes {
            let len = chunk.min(bytes - off);
            let st = &mut ring[i % 3];
            self.wait_mapped(st)?;
            {
                let mut view = st
                    .buf
                    .get_mapped_range_mut(..len as u64)
                    .map_err(|e| err("écriture d'un tampon", e))?;
                view.copy_from_slice(unsafe { std::slice::from_raw_parts(src.add(off), len) });
            }
            st.buf.unmap();
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&st.buf, 0, &dst.buf, (dst_off + off) as u64, len as u64);
            let idx = self.queue.submit([enc.finish()]);
            self.remap(st, wgpu::MapMode::Write, &idx);
            off += len;
            i += 1;
        }
        for st in ring {
            self.recycle(st, true);
        }
        Ok(())
    }

    /// Rapatrie `bytes` octets de `src` (à partir de `src_off`) vers l'hôte, par blocs (anneau de 3).
    ///
    /// # Safety
    /// `dst` doit pointer vers `bytes` octets accessibles en écriture.
    pub unsafe fn download(
        &self,
        src: &WgBuffer,
        src_off: usize,
        dst: *mut u8,
        bytes: usize,
        chunk: usize,
    ) -> Result<(), BackendError> {
        self.check()?;
        let chunk = chunk.min(bytes).max(4);
        let mut ring: Vec<Staging> = (0..3).map(|_| self.staging(chunk, false)).collect();
        let mut pending: std::collections::VecDeque<(usize, usize, usize)> = Default::default(); // (slot, off, len)
        let mut off = 0;
        let mut i = 0;
        loop {
            // Termine le plus ancien bloc quand l'anneau est plein (ou à la fin).
            if pending.len() == 3 || (off >= bytes && !pending.is_empty()) {
                let (slot, o, len) = pending.pop_front().unwrap();
                let st = &ring[slot];
                self.wait_mapped(st)?;
                {
                    let view = st
                        .buf
                        .get_mapped_range(..len as u64)
                        .map_err(|e| err("lecture d'un tampon", e))?;
                    unsafe { std::ptr::copy_nonoverlapping(view.as_ptr(), dst.add(o), len) };
                }
                st.buf.unmap();
                continue;
            }
            if off >= bytes {
                break;
            }
            let len = chunk.min(bytes - off);
            let slot = i % 3;
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(
                &src.buf,
                (src_off + off) as u64,
                &ring[slot].buf,
                0,
                len.div_ceil(4) as u64 * 4,
            );
            let idx = self.queue.submit([enc.finish()]);
            self.remap(&mut ring[slot], wgpu::MapMode::Read, &idx);
            pending.push_back((slot, off, len));
            off += len;
            i += 1;
        }
        for st in ring {
            self.recycle(st, false);
        }
        Ok(())
    }

    /// Exécute `k` en mode Flux : bloc par bloc, envoi du bloc k+1 et retour du bloc k−1 pendant le
    /// calcul du bloc k ; chaque bloc de chaque sortie est écrit directement dans sa destination
    /// (`outs` : (adresse hôte, octets par élément)).
    ///
    /// # Safety
    /// Les pointeurs hôte (entrées `Host`, destinations) doivent couvrir `n` éléments.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn flux(
        &self,
        k: &WgKernel,
        inputs: &[FluxIn],
        outs: &[(*mut u8, usize)],
        scalars: &[u32],
        fparams: &[f64],
        n: usize,
        chunk_elems: usize,
    ) -> Result<(), BackendError> {
        self.check()?;
        const SLOTS: usize = 3;
        let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let r =
            unsafe { self.flux_inner(k, inputs, outs, scalars, fparams, n, chunk_elems, SLOTS) };
        let errors = [
            pollster::block_on(validation.pop()),
            pollster::block_on(oom.pop()),
        ];
        r?;
        if let Some(e) = errors.into_iter().flatten().next() {
            return Err(BackendError(format!(
                "wgpu : traitement par blocs refusé sur {} ({e})",
                self.info.name
            )));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn flux_inner(
        &self,
        k: &WgKernel,
        inputs: &[FluxIn],
        outs: &[(*mut u8, usize)],
        scalars: &[u32],
        fparams: &[f64],
        n: usize,
        chunk_elems: usize,
        slots: usize,
    ) -> Result<(), BackendError> {
        struct Slot {
            ins: Vec<WgBuffer>,
            ups: Vec<Option<Staging>>,
            outs: Vec<WgBuffer>,
            backs: Vec<Staging>,
            busy: Option<(usize, usize)>, // (élément de départ, nombre d'éléments)
        }
        let chunk_elems = chunk_elems.max(1).min(n);
        let mut ring: Vec<Slot> = Vec::with_capacity(slots);
        for _ in 0..slots.min(n.div_ceil(chunk_elems)) {
            let mut ins = Vec::new();
            let mut ups = Vec::new();
            for i in inputs {
                let bytes = chunk_elems * i.elem();
                ins.push(self.alloc(bytes)?);
                ups.push(matches!(i, FluxIn::Host { .. }).then(|| self.staging(bytes, true)));
            }
            let mut obufs = Vec::new();
            let mut backs = Vec::new();
            for (_, elem) in outs {
                obufs.push(self.alloc(chunk_elems * elem)?);
                backs.push(self.staging(chunk_elems * elem, false));
            }
            ring.push(Slot {
                ins,
                ups,
                outs: obufs,
                backs,
                busy: None,
            });
        }
        let finish = |dev: &WgDevice, s: &mut Slot| -> Result<(), BackendError> {
            if let Some((start, len)) = s.busy.take() {
                for (back, (dst, elem)) in s.backs.iter().zip(outs) {
                    dev.wait_mapped(back)?;
                    let bytes = len * elem;
                    {
                        let view = back
                            .buf
                            .get_mapped_range(..bytes as u64)
                            .map_err(|e| err("lecture d'un tampon", e))?;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                view.as_ptr(),
                                dst.add(start * elem),
                                bytes,
                            )
                        };
                    }
                    back.buf.unmap();
                }
            }
            Ok(())
        };
        let mut start = 0;
        let mut i = 0;
        while start < n {
            let len = chunk_elems.min(n - start);
            let slot = i % ring.len();
            finish(self, &mut ring[slot])?;
            let s = &mut ring[slot];
            let mut enc = self.device.create_command_encoder(&Default::default());
            for (a, input) in inputs.iter().enumerate() {
                let bytes = len * input.elem();
                match input {
                    FluxIn::Host { ptr, elem } => {
                        let st = s.ups[a].as_mut().unwrap();
                        self.wait_mapped(st)?;
                        {
                            let mut view = st
                                .buf
                                .get_mapped_range_mut(..bytes as u64)
                                .map_err(|e| err("écriture d'un tampon", e))?;
                            view.copy_from_slice(unsafe {
                                std::slice::from_raw_parts(ptr.add(start * elem), bytes)
                            });
                        }
                        st.buf.unmap();
                        enc.copy_buffer_to_buffer(&st.buf, 0, &s.ins[a].buf, 0, bytes as u64);
                    }
                    FluxIn::Dev { buf, elem } => {
                        enc.copy_buffer_to_buffer(
                            &buf.buf,
                            (start * elem) as u64,
                            &s.ins[a].buf,
                            0,
                            bytes as u64,
                        );
                    }
                }
            }
            let mut params = scalars.to_vec();
            params.push(len as u32);
            let ins: Vec<&WgBuffer> = s.ins.iter().collect();
            let obufs: Vec<&WgBuffer> = s.outs.iter().collect();
            self.encode(&mut enc, k, &ins, &obufs, &params, fparams, len)?;
            for ((ob, back), (_, elem)) in s.outs.iter().zip(&s.backs).zip(outs) {
                enc.copy_buffer_to_buffer(
                    &ob.buf,
                    0,
                    &back.buf,
                    0,
                    (len * elem).div_ceil(4) as u64 * 4,
                );
            }
            let idx = self.queue.submit([enc.finish()]);
            for st in s.ups.iter_mut().flatten() {
                self.remap(st, wgpu::MapMode::Write, &idx);
            }
            for back in s.backs.iter_mut() {
                self.remap(back, wgpu::MapMode::Read, &idx);
            }
            s.busy = Some((start, len));
            start += len;
            i += 1;
        }
        // Termine les blocs encore en vol, dans l'ordre.
        let used = ring.len();
        for j in 0..used {
            let slot = (i + j) % used;
            finish(self, &mut ring[slot])?;
        }
        for s in ring {
            for st in s.ups.into_iter().flatten() {
                self.recycle(st, true);
            }
            for back in s.backs {
                self.recycle(back, false);
            }
        }
        Ok(())
    }

    pub fn finish(&self) -> Result<(), BackendError> {
        self.check()?;
        let r = self.device.poll(wgpu::PollType::wait_indefinitely());
        self.check()?;
        r.map_err(|e| err("attente du GPU", e))?;
        Ok(())
    }
}

/// Tampon de transfert (mappable), avec son état de mappage.
struct Staging {
    buf: wgpu::Buffer,
    bytes: usize,
    mapped: Arc<AtomicBool>,
    submission: Option<wgpu::SubmissionIndex>,
}

/// Entrée d'un traitement par blocs.
pub enum FluxIn<'a> {
    /// Tableau en mémoire de l'hôte (`elem` octets par élément).
    Host { ptr: *const u8, elem: usize },
    /// Buffer déjà sur le périphérique (résultat d'un kernel précédent).
    Dev { buf: &'a WgBuffer, elem: usize },
}

impl FluxIn<'_> {
    fn elem(&self) -> usize {
        match self {
            FluxIn::Host { elem, .. } | FluxIn::Dev { elem, .. } => *elem,
        }
    }
}

fn bytemuck_u32(words: &[u32]) -> &[u8] {
    // u32 → octets (ordre natif), sans dépendance supplémentaire.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), std::mem::size_of_val(words)) }
}
