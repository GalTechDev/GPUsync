//! Exécution wgpu : un `Device` par adaptateur, pipelines en cache, buffers de stockage.
//!
//! Disposition des bindings (voir gpusync-codegen) : tableaux d'entrée, sortie, puis `gs_params`
//! (scalaires et `n`, un mot de 32 bits chacun). La grille est 2D pour dépasser 65 535 groupes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use gpusync_runtime::{BackendError, DeviceInfo};
use wgpu::util::DeviceExt;

pub const WORKGROUP_SIZE: u32 = 256;

pub struct WgDevice {
    pub info: DeviceInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    max_groups_per_dim: u32,
    kernels: Mutex<HashMap<String, Arc<WgKernel>>>,
}

pub struct WgKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    n_inputs: usize,
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
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| err("ouverture du périphérique", e))?;
        // Par défaut wgpu panique sur une erreur non capturée : on la journalise plutôt.
        device.on_uncaptured_error(Arc::new(|e| {
            eprintln!("[gpusync] erreur wgpu non capturée : {e}")
        }));
        let d = Arc::new(WgDevice {
            info: info.clone(),
            device,
            queue,
            max_groups_per_dim: limits.max_compute_workgroups_per_dimension.max(1),
            kernels: Mutex::new(HashMap::new()),
        });
        open.insert(key, d.clone());
        Ok(d)
    }

    /// Compile (avec cache) un module WGSL dont le point d'entrée est `main`.
    pub fn compile(&self, code: &str, n_inputs: usize) -> Result<Arc<WgKernel>, BackendError> {
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
        entries.push(storage(n_inputs as u32, false));
        entries.push(storage(n_inputs as u32 + 1, true));
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
        });
        cache.insert(code.to_string(), k.clone());
        Ok(k)
    }

    pub fn alloc(&self, bytes: usize) -> Result<WgBuffer, BackendError> {
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
        out: &WgBuffer,
        params: &[u32],
        n: usize,
    ) -> Result<(), BackendError> {
        if inputs.len() != k.n_inputs {
            return Err(BackendError(
                "nombre de tableaux d'entrée incohérent".into(),
            ));
        }
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck_u32(params),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let mut entries: Vec<wgpu::BindGroupEntry> = inputs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.buf.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: out.buf.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32 + 1,
            resource: params_buf.as_entire_binding(),
        });
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
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(gx, gy as u32, 1);
        }
        self.queue.submit([enc.finish()]);
        Ok(())
    }

    /// Copie bloquante périphérique → hôte (attend la fin des calculs soumis).
    pub fn read(&self, buf: &WgBuffer, out: &mut [u8]) -> Result<(), BackendError> {
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

    pub fn finish(&self) -> Result<(), BackendError> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| err("attente du GPU", e))?;
        Ok(())
    }
}

fn bytemuck_u32(words: &[u32]) -> &[u8] {
    // u32 → octets (ordre natif), sans dépendance supplémentaire.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), std::mem::size_of_val(words)) }
}
