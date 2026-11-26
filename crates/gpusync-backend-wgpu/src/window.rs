//! Native GPU window (`gs.Window`) based on `winit` + `wgpu::Surface`.
//!
//! When a `gs.Array` (RGBA8 packed in `u32`, shape `(H, W)`) already lives on the same
//! wgpu device (Metal, Vulkan, DX12), `present_wg` reads its `wgpu::Buffer` directly in a
//! full-screen shader and shows it in the swapchain — **zero GPU → CPU copy**. For other
//! backends (OpenCL, CUDA, CPU) or NumPy arrays, `present_bytes` uploads the bytes through a
//! reused GPU buffer.

use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use gpusync_runtime::BackendError;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent as WinitEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::platform::pump_events::EventLoopExtPumpEvents;
use winit::window::{Window, WindowAttributes, WindowId};

use crate::exec::{WgBuffer, WgDevice};

/// Discrete event returned by `WgWindow::poll()`.
#[derive(Debug, Clone)]
pub enum WindowEvent {
    Close,
    Resize { width: u32, height: u32 },
    KeyDown { key: String },
    KeyUp { key: String },
    MouseDown { button: String },
    MouseUp { button: String },
    MouseMove { x: f64, y: f64, dx: f64, dy: f64 },
    MouseWheel { dx: f64, dy: f64 },
}

const BLIT_WGSL: &str = r#"
struct Params {
    fb_w: u32,
    fb_h: u32,
    surf_w: u32,
    surf_h: u32,
    is_srgb: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<uniform> p: Params;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i & 1u) * 4u) - 1.0;
    let y = f32((i & 2u) * 2u) - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px = min(u32(pos.x * f32(p.fb_w) / f32(max(p.surf_w, 1u))), p.fb_w - 1u);
    let py = min(u32(pos.y * f32(p.fb_h) / f32(max(p.surf_h, 1u))), p.fb_h - 1u);
    let c = pixels[py * p.fb_w + px];
    var r = f32(c & 255u) / 255.0;
    var g = f32((c >> 8u) & 255u) / 255.0;
    var b = f32((c >> 16u) & 255u) / 255.0;
    if (p.is_srgb != 0u) {
        r = pow(r, 2.2);
        g = pow(g, 2.2);
        b = pow(b, 2.2);
    }
    return vec4<f32>(r, g, b, 1.0);
}
"#;

// `EventLoop::new()` can only be called once per process in winit: it is reused
// across successive windows (notebooks, scripts, tests) on the main thread.
thread_local! {
    static EVENT_LOOP: RefCell<Option<EventLoop<()>>> = const { RefCell::new(None) };
}

fn take_event_loop() -> Result<EventLoop<()>, BackendError> {
    EVENT_LOOP.with(|slot| {
        if let Some(el) = slot.borrow_mut().take() {
            return Ok(el);
        }
        EventLoop::new()
            .map_err(|e| BackendError(format!("cannot create the window event loop ({e})")))
    })
}

fn return_event_loop(el: EventLoop<()>) {
    EVENT_LOOP.with(|slot| {
        *slot.borrow_mut() = Some(el);
    });
}

struct AppState {
    pending_attrs: Option<WindowAttributes>,
    window: Option<Arc<Window>>,
    open: bool,
    logical_size: (u32, u32),
    physical_size: (u32, u32),
    resized: bool,
    events: Vec<WindowEvent>,
    keys_down: HashSet<String>,
    mouse_buttons: HashSet<String>,
    mouse_pos: (f64, f64),
    has_mouse_pos: bool,
    create_err: Option<String>,
}

impl AppState {
    fn ensure_window(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            if let Some(attrs) = self.pending_attrs.take() {
                match event_loop.create_window(attrs) {
                    Ok(win) => {
                        let scale = win.scale_factor();
                        let phys = win.inner_size();
                        let log: LogicalSize<u32> = phys.to_logical(scale);
                        self.physical_size = (phys.width.max(1), phys.height.max(1));
                        self.logical_size = (log.width.max(1), log.height.max(1));
                        self.window = Some(Arc::new(win));
                    }
                    Err(e) => {
                        self.create_err = Some(e.to_string());
                        self.open = false;
                    }
                }
            }
        }
    }
}

impl ApplicationHandler for AppState {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.ensure_window(event_loop);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.ensure_window(event_loop);
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WinitEvent,
    ) {
        match event {
            WinitEvent::CloseRequested => {
                self.open = false;
                self.events.push(WindowEvent::Close);
            }
            WinitEvent::Resized(phys) => {
                if phys.width > 0 && phys.height > 0 {
                    let scale = self
                        .window
                        .as_ref()
                        .map(|w| w.scale_factor())
                        .unwrap_or(1.0);
                    let log: LogicalSize<u32> = phys.to_logical(scale);
                    self.physical_size = (phys.width, phys.height);
                    self.logical_size = (log.width.max(1), log.height.max(1));
                    self.resized = true;
                    self.events.push(WindowEvent::Resize {
                        width: self.logical_size.0,
                        height: self.logical_size.1,
                    });
                }
            }
            WinitEvent::ScaleFactorChanged { .. } => {
                if let Some(win) = &self.window {
                    let phys = win.inner_size();
                    if phys.width > 0 && phys.height > 0 {
                        let log: LogicalSize<u32> = phys.to_logical(win.scale_factor());
                        self.physical_size = (phys.width, phys.height);
                        self.logical_size = (log.width.max(1), log.height.max(1));
                        self.resized = true;
                        self.events.push(WindowEvent::Resize {
                            width: self.logical_size.0,
                            height: self.logical_size.1,
                        });
                    }
                }
            }
            WinitEvent::KeyboardInput { event, .. } => {
                if let Some(name) = key_name(&event.logical_key) {
                    match event.state {
                        ElementState::Pressed if !event.repeat => {
                            self.keys_down.insert(name.clone());
                            self.events.push(WindowEvent::KeyDown { key: name });
                        }
                        ElementState::Released => {
                            self.keys_down.remove(&name);
                            self.events.push(WindowEvent::KeyUp { key: name });
                        }
                        _ => {}
                    }
                }
            }
            WinitEvent::MouseInput { state, button, .. } => {
                let b = match button {
                    MouseButton::Left => "left",
                    MouseButton::Right => "right",
                    MouseButton::Middle => "middle",
                    _ => return,
                }
                .to_string();
                match state {
                    ElementState::Pressed => {
                        self.mouse_buttons.insert(b.clone());
                        self.events.push(WindowEvent::MouseDown { button: b });
                    }
                    ElementState::Released => {
                        self.mouse_buttons.remove(&b);
                        self.events.push(WindowEvent::MouseUp { button: b });
                    }
                }
            }
            WinitEvent::CursorMoved { position, .. } => {
                let scale = self
                    .window
                    .as_ref()
                    .map(|w| w.scale_factor())
                    .unwrap_or(1.0);
                let x = position.x / scale;
                let y = position.y / scale;
                let (dx, dy) = if self.has_mouse_pos {
                    (x - self.mouse_pos.0, y - self.mouse_pos.1)
                } else {
                    (0.0, 0.0)
                };
                self.mouse_pos = (x, y);
                self.has_mouse_pos = true;
                self.events.push(WindowEvent::MouseMove { x, y, dx, dy });
            }
            WinitEvent::MouseWheel { delta, .. } => {
                let (dx, dy) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x as f64, y as f64),
                    MouseScrollDelta::PixelDelta(p) => (p.x / 24.0, p.y / 24.0),
                };
                self.events.push(WindowEvent::MouseWheel { dx, dy });
            }
            _ => {}
        }
    }
}

fn key_name(key: &Key) -> Option<String> {
    match key {
        Key::Character(s) => Some(s.to_lowercase()),
        Key::Named(n) => Some(
            match n {
                NamedKey::Space => "space",
                NamedKey::Escape => "escape",
                NamedKey::Enter => "enter",
                NamedKey::Tab => "tab",
                NamedKey::Backspace => "backspace",
                NamedKey::ArrowUp => "up",
                NamedKey::ArrowDown => "down",
                NamedKey::ArrowLeft => "left",
                NamedKey::ArrowRight => "right",
                NamedKey::Shift => "shift",
                NamedKey::Control => "ctrl",
                NamedKey::Alt => "alt",
                NamedKey::Super => "super",
                _ => return None,
            }
            .to_string(),
        ),
        _ => None,
    }
}

/// Native wgpu + winit window (must live on the main thread).
pub struct WgWindow {
    event_loop: Option<EventLoop<()>>,
    app: AppState,
    dev: Arc<WgDevice>,
    surface: Option<wgpu::Surface<'static>>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    upload_buf: Option<wgpu::Buffer>,
    vsync: bool,
    is_srgb: bool,
}

impl WgWindow {
    /// Opens a native window and attaches it to the wgpu device (`dev` or the first available wgpu GPU).
    pub fn open(
        width: u32,
        height: u32,
        title: &str,
        vsync: bool,
        resizable: bool,
        dev: Option<Arc<WgDevice>>,
    ) -> Result<Self, BackendError> {
        let st = crate::state()?;
        let dev = match dev {
            Some(d) => d,
            None => {
                let first = st.devices.first().ok_or_else(|| {
                    BackendError("no wgpu GPU available to open a window".into())
                })?;
                WgDevice::open(first.1.backend, first.1.index)?
            }
        };

        let mut event_loop = take_event_loop()?;
        let attrs = WindowAttributes::default()
            .with_title(title)
            .with_inner_size(LogicalSize::new(width.max(1), height.max(1)))
            .with_resizable(resizable);

        let mut app = AppState {
            pending_attrs: Some(attrs),
            window: None,
            open: true,
            logical_size: (width.max(1), height.max(1)),
            physical_size: (width.max(1), height.max(1)),
            resized: false,
            events: Vec::new(),
            keys_down: HashSet::new(),
            mouse_buttons: HashSet::new(),
            mouse_pos: (0.0, 0.0),
            has_mouse_pos: false,
            create_err: None,
        };

        // Triggers `resumed` to create the native window on the main thread.
        event_loop.pump_app_events(Some(Duration::ZERO), &mut app);
        if let Some(e) = app.create_err.take() {
            return_event_loop(event_loop);
            return Err(BackendError(format!("window creation: {e}")));
        }
        let Some(window) = app.window.clone() else {
            return_event_loop(event_loop);
            return Err(BackendError(
                "winit could not initialize the native window".into(),
            ));
        };

        let surface = match st.instance.create_surface(window) {
            Ok(s) => s,
            Err(e) => {
                app.window = None;
                return_event_loop(event_loop);
                return Err(BackendError(format!("wgpu: surface creation ({e})")));
            }
        };

        let adapter = &st.adapters[dev.adapter_idx];
        let caps = surface.get_capabilities(adapter);
        // Prefer a linear format (Bgra8Unorm / Rgba8Unorm): the u32 pixels are already sRGB.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| BackendError("wgpu surface without a compatible format".into()))?;
        let is_srgb = format.is_srgb();

        let present_mode = pick_present_mode(&caps.present_modes, vsync);
        let alpha_mode = caps
            .alpha_modes
            .first()
            .copied()
            .unwrap_or(wgpu::CompositeAlphaMode::Auto);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: app.physical_size.0.max(1),
            height: app.physical_size.1.max(1),
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&dev.device, &config);

        let shader = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpusync-window-blit"),
                source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
            });

        let bind_layout = dev
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpusync-window-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = dev
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpusync-window-pl"),
                bind_group_layouts: &[Some(&bind_layout)],
                immediate_size: 0,
            });

        let pipeline = dev
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gpusync-window-pipe"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let params_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpusync-window-params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            event_loop: Some(event_loop),
            app,
            dev,
            surface: Some(surface),
            config,
            pipeline,
            bind_layout,
            params_buf,
            upload_buf: None,
            vsync,
            is_srgb,
        })
    }

    pub fn device(&self) -> &Arc<WgDevice> {
        &self.dev
    }

    pub fn device_id(&self) -> String {
        self.dev.info.id()
    }

    pub fn is_open(&self) -> bool {
        self.app.open
    }

    pub fn size(&self) -> (u32, u32) {
        self.app.logical_size
    }

    pub fn physical_size(&self) -> (u32, u32) {
        self.app.physical_size
    }

    pub fn vsync(&self) -> bool {
        self.vsync
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        if self.vsync == vsync {
            return;
        }
        self.vsync = vsync;
        if let (Some(surface), Ok(st)) = (&self.surface, crate::state()) {
            let caps = surface.get_capabilities(&st.adapters[self.dev.adapter_idx]);
            self.config.present_mode = pick_present_mode(&caps.present_modes, vsync);
            surface.configure(&self.dev.device, &self.config);
        }
    }

    pub fn set_title(&self, title: &str) {
        if let Some(win) = &self.app.window {
            win.set_title(title);
        }
    }

    pub fn keys_down(&self) -> &HashSet<String> {
        &self.app.keys_down
    }

    pub fn mouse_buttons(&self) -> &HashSet<String> {
        &self.app.mouse_buttons
    }

    pub fn mouse_pos(&self) -> (f64, f64) {
        self.app.mouse_pos
    }

    /// Pumps system events (non-blocking) and returns those gathered since the last call.
    pub fn poll(&mut self) -> Vec<WindowEvent> {
        if let Some(el) = &mut self.event_loop {
            el.pump_app_events(Some(Duration::ZERO), &mut self.app);
        }
        if self.app.resized {
            self.reconfigure();
        }
        std::mem::take(&mut self.app.events)
    }

    fn reconfigure(&mut self) {
        self.app.resized = false;
        let (pw, ph) = self.app.physical_size;
        if pw > 0 && ph > 0 {
            self.config.width = pw;
            self.config.height = ph;
            if let Some(surface) = &self.surface {
                surface.configure(&self.dev.device, &self.config);
            }
        }
    }

    /// Shows a `WgBuffer` directly (RGBA8 packed in `u32`, `width × height`): **zero CPU copy**.
    pub fn present_wg(
        &mut self,
        buf: &WgBuffer,
        width: u32,
        height: u32,
    ) -> Result<(), BackendError> {
        if !self.app.open || width == 0 || height == 0 {
            return Ok(());
        }
        let need = (width as usize) * (height as usize) * 4;
        if buf.bytes < need {
            return Err(BackendError(format!(
                "gs.Window.present: buffer too small ({} B for {width}×{height} u32)",
                buf.bytes
            )));
        }
        if self.app.resized {
            self.reconfigure();
        }
        let _ = self.dev.flush_batch();
        self.dev.check_buf(buf)?;
        self.blit_buffer(&buf.buf, width, height)
    }

    /// Shows host bytes (RGBA8 / packed u32) through a reused GPU buffer.
    pub fn present_bytes(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), BackendError> {
        if !self.app.open || width == 0 || height == 0 {
            return Ok(());
        }
        let need = (width as usize) * (height as usize) * 4;
        if rgba.len() < need {
            return Err(BackendError(format!(
                "gs.Window.present: {} bytes received for {width}×{height} RGBA",
                rgba.len()
            )));
        }
        if self.app.resized {
            self.reconfigure();
        }
        let need_u64 = (need as u64).max(4);
        let recreate = self.upload_buf.as_ref().is_none_or(|b| b.size() < need_u64);
        if recreate {
            self.upload_buf = Some(self.dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpusync-window-upload"),
                size: need_u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let buf = self.upload_buf.as_ref().unwrap().clone();
        self.dev.queue.write_buffer(&buf, 0, &rgba[..need]);
        self.blit_buffer(&buf, width, height)
    }

    fn blit_buffer(
        &mut self,
        pixels: &wgpu::Buffer,
        fb_w: u32,
        fb_h: u32,
    ) -> Result<(), BackendError> {
        let _ = self.dev.flush_batch(); // a rejected batch only affects its own results
        let Some(surface) = &self.surface else {
            return Ok(());
        };
        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) => f,
            wgpu::CurrentSurfaceTexture::Suboptimal(f) => {
                self.app.resized = true;
                f
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.reconfigure();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(BackendError(
                    "wgpu: validation error on the surface".into(),
                ));
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let params: [u32; 8] = [
            fb_w,
            fb_h,
            self.config.width.max(1),
            self.config.height.max(1),
            u32::from(self.is_srgb),
            0,
            0,
            0,
        ];
        let pbytes = unsafe {
            std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(&params))
        };
        self.dev.queue.write_buffer(&self.params_buf, 0, pbytes);

        let bind = self
            .dev
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bind_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pixels.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.params_buf.as_entire_binding(),
                    },
                ],
            });

        let mut enc = self
            .dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::DontCare(unsafe { wgpu::LoadOpDontCare::enabled() }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }

        self.dev.queue.submit([enc.finish()]);
        if let Some(win) = &self.app.window {
            win.pre_present_notify();
        }
        self.dev.queue.present(frame);
        Ok(())
    }

    pub fn close(&mut self) {
        self.app.open = false;
        self.surface = None;
        self.app.window = None;
        if let Some(el) = &mut self.event_loop {
            el.pump_app_events(Some(Duration::ZERO), &mut self.app);
        }
        if let Some(el) = self.event_loop.take() {
            return_event_loop(el);
        }
    }
}

impl Drop for WgWindow {
    fn drop(&mut self) {
        self.close();
    }
}

fn pick_present_mode(modes: &[wgpu::PresentMode], vsync: bool) -> wgpu::PresentMode {
    if vsync {
        return wgpu::PresentMode::Fifo;
    }
    for wanted in [
        wgpu::PresentMode::Immediate,
        wgpu::PresentMode::Mailbox,
        wgpu::PresentMode::FifoRelaxed,
        wgpu::PresentMode::Fifo,
    ] {
        if modes.contains(&wanted) {
            return wanted;
        }
    }
    wgpu::PresentMode::AutoNoVsync
}
