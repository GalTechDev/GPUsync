//! Backend wgpu : Metal (macOS), Vulkan et DX12 (Windows, Linux).
//!
//! Un même GPU apparaît souvent via plusieurs API (Vulkan et DX12 sous Windows) : on n'en garde
//! qu'une entrée par GPU physique, dans l'ordre de préférence Metal > Vulkan > DX12.
//! Les adaptateurs logiciels (CPU) sont ignorés : le CPU passe par OpenCL (POCL) ou l'interpréteur.

use gpusync_runtime::{Backend, BackendError, DeviceInfo, DeviceKind};

pub struct WgpuBackend;

/// Ce que wgpu nous apprend d'un adaptateur (séparé de wgpu pour tester le tri sans GPU).
#[derive(Debug, Clone)]
struct Candidate {
    api: &'static str,
    name: String,
    vendor_id: u32,
    device_id: u32,
    kind: DeviceKind,
    driver: String,
    f64: bool,
    max_buffer_bytes: u64,
}

impl Backend for WgpuBackend {
    fn name(&self) -> &'static str {
        "wgpu"
    }

    fn devices(&self) -> Result<Vec<DeviceInfo>, BackendError> {
        let backends = wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = pollster::block_on(instance.enumerate_adapters(backends));
        if adapters.is_empty() {
            return Err(BackendError(
                "aucun adaptateur Metal, Vulkan ou DX12".into(),
            ));
        }
        let candidates = adapters
            .iter()
            .filter_map(|a| {
                let info = a.get_info();
                let api = match info.backend {
                    wgpu::Backend::Metal => "metal",
                    wgpu::Backend::Vulkan => "vulkan",
                    wgpu::Backend::Dx12 => "dx12",
                    _ => return None,
                };
                let driver = [info.driver.trim(), info.driver_info.trim()]
                    .iter()
                    .filter(|s| !s.is_empty())
                    .copied()
                    .collect::<Vec<_>>()
                    .join(" ");
                Some(Candidate {
                    api,
                    name: info.name.trim().to_string(),
                    vendor_id: info.vendor,
                    device_id: info.device,
                    kind: match info.device_type {
                        wgpu::DeviceType::DiscreteGpu => DeviceKind::DiscreteGpu,
                        wgpu::DeviceType::IntegratedGpu => DeviceKind::IntegratedGpu,
                        wgpu::DeviceType::VirtualGpu => DeviceKind::VirtualGpu,
                        wgpu::DeviceType::Cpu => DeviceKind::Cpu,
                        wgpu::DeviceType::Other => DeviceKind::Other,
                    },
                    driver,
                    f64: a.features().contains(wgpu::Features::SHADER_F64),
                    max_buffer_bytes: a.limits().max_buffer_size,
                })
            })
            .collect();
        Ok(select(candidates))
    }
}

fn api_rank(api: &str) -> u8 {
    match api {
        "metal" => 0,
        "vulkan" => 1,
        _ => 2,
    }
}

/// Garde un adaptateur par GPU physique (API préférée), ignore le CPU, numérote par API.
fn select(mut candidates: Vec<Candidate>) -> Vec<DeviceInfo> {
    candidates.retain(|c| c.kind != DeviceKind::Cpu);
    candidates.sort_by_key(|c| api_rank(c.api));
    let mut seen: Vec<(u32, u32, String)> = Vec::new();
    let mut out: Vec<DeviceInfo> = Vec::new();
    for c in candidates {
        let key = (c.vendor_id, c.device_id, c.name.to_lowercase());
        // Certains drivers rapportent des identifiants nuls : le nom départage alors.
        let duplicate = seen
            .iter()
            .any(|k| (k.0 == key.0 && k.1 == key.1 && key.1 != 0) || k.2 == key.2);
        if duplicate {
            continue;
        }
        seen.push(key);
        let index = out.iter().filter(|d| d.backend == c.api).count();
        // Metal ne rapporte ni fabricant ni driver : seul Apple fournit Metal.
        let vendor = match (vendor_name(c.vendor_id), c.api) {
            ("", "metal") => "Apple",
            (v, _) => v,
        };
        let version = if c.driver.is_empty() {
            c.api.to_string()
        } else {
            c.driver
        };
        out.push(DeviceInfo {
            name: c.name,
            backend: c.api,
            index,
            kind: c.kind,
            vendor: vendor.to_string(),
            version,
            f64: c.f64,
            unified_memory: c.kind == DeviceKind::IntegratedGpu,
            memory_bytes: None,
            max_buffer_bytes: Some(c.max_buffer_bytes),
        });
    }
    out
}

fn vendor_name(id: u32) -> &'static str {
    match id {
        0x10DE => "NVIDIA",
        0x1002 | 0x1022 => "AMD",
        0x8086 => "Intel",
        0x106B => "Apple",
        0x5143 => "Qualcomm",
        0x13B5 => "ARM",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(api: &'static str, name: &str, dev: u32, kind: DeviceKind) -> Candidate {
        Candidate {
            api,
            name: name.into(),
            vendor_id: 0x10DE,
            device_id: dev,
            kind,
            driver: String::new(),
            f64: api == "vulkan",
            max_buffer_bytes: 1 << 30,
        }
    }

    #[test]
    fn un_gpu_vu_par_vulkan_et_dx12_n_apparait_qu_une_fois_en_vulkan() {
        let devs = select(vec![
            cand("dx12", "RTX", 1, DeviceKind::DiscreteGpu),
            cand("vulkan", "RTX", 1, DeviceKind::DiscreteGpu),
            cand("dx12", "Intel UHD", 2, DeviceKind::IntegratedGpu),
            cand("vulkan", "Intel UHD", 2, DeviceKind::IntegratedGpu),
            cand("dx12", "Microsoft Basic Render Driver", 3, DeviceKind::Cpu),
        ]);
        let ids: Vec<_> = devs.iter().map(|d| (d.id(), d.name.as_str())).collect();
        assert_eq!(
            ids,
            vec![("vulkan:0".into(), "RTX"), ("vulkan:1".into(), "Intel UHD")]
        );
        assert!(devs[1].unified_memory && !devs[0].unified_memory);
    }

    #[test]
    fn la_decouverte_ne_plante_jamais() {
        let _ = WgpuBackend.devices();
    }
}
