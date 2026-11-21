//! Backend OpenCL (≥ 1.2). La bibliothèque OpenCL est chargée dynamiquement au premier appel :
//! si elle est absente, le backend est simplement indisponible (aucune erreur à l'import).

mod exec;

pub use exec::{ClArg, ClBuffer, ClDevice, ClKernel, Pinned, wait};
pub use opencl3::event::Event as ClEvent;

use gpusync_runtime::{Backend, BackendError, DeviceInfo, DeviceKind};
use opencl3::device::{
    CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, Device, cl_device_id,
    cl_device_type,
};
use opencl3::platform::get_platforms;

pub struct OpenClBackend;

impl Backend for OpenClBackend {
    fn name(&self) -> &'static str {
        "opencl"
    }

    fn devices(&self) -> Result<Vec<DeviceInfo>, BackendError> {
        Ok(enumerate()?.into_iter().map(|(_, info)| info).collect())
    }
}

/// Périphériques OpenCL, dans un ordre stable (plateformes puis périphériques) : l'index de
/// `DeviceInfo` sert à rouvrir le même périphérique.
pub(crate) fn enumerate() -> Result<Vec<(cl_device_id, DeviceInfo)>, BackendError> {
    if let Err(e) = cl3::load_library() {
        return Err(BackendError(format!(
            "bibliothèque OpenCL introuvable ({e})"
        )));
    }
    let platforms =
        get_platforms().map_err(|e| BackendError(format!("aucune plateforme OpenCL ({e})")))?;
    let mut out = Vec::new();
    for platform in platforms {
        let platform_name = platform.name().unwrap_or_default();
        let ids = match platform.get_devices(CL_DEVICE_TYPE_ALL) {
            Ok(ids) => ids,
            Err(_) => continue, // plateforme sans périphérique : on l'ignore
        };
        for id in ids {
            let d = Device::new(id);
            let unified = d.host_unified_memory().unwrap_or(false);
            let info = DeviceInfo {
                name: d
                    .name()
                    .unwrap_or_else(|_| platform_name.clone())
                    .trim()
                    .to_string(),
                backend: "opencl",
                index: out.len(),
                kind: kind_of(d.dev_type().unwrap_or(0), unified),
                vendor: d.vendor().unwrap_or_default().trim().to_string(),
                version: d.version().unwrap_or_default().trim().to_string(),
                f64: d.double_fp_config().map(|c| c != 0).unwrap_or(false),
                // OpenCL fournit toute la bibliothèque mathématique en double précision.
                f64_math: d.double_fp_config().map(|c| c != 0).unwrap_or(false),
                // `long` / `ulong` font partie d'OpenCL C (profil complet), sauf profil embarqué.
                int64: !d.profile().unwrap_or_default().contains("EMBEDDED"),
                unified_memory: unified,
                memory_bytes: d.global_mem_size().ok(),
                max_buffer_bytes: d.max_mem_alloc_size().ok(),
            };
            out.push((id, info));
        }
    }
    // L'ordre des plateformes donné par le chargeur OpenCL n'est pas stable d'un environnement à
    // l'autre (constaté sous Windows : RTX puis UHD 770, puis l'inverse) : on trie, pour que
    // `opencl:0` désigne toujours le même GPU.
    out.sort_by(|(_, a), (_, b)| {
        (a.kind.rank(), &a.name, &a.vendor).cmp(&(b.kind.rank(), &b.name, &b.vendor))
    });
    for (i, (_, info)) in out.iter_mut().enumerate() {
        info.index = i;
    }
    Ok(out)
}

/// OpenCL ne distingue pas GPU intégré / dédié : la mémoire unifiée sert d'indice.
fn kind_of(dev_type: cl_device_type, unified: bool) -> DeviceKind {
    if dev_type & CL_DEVICE_TYPE_GPU != 0 {
        if unified {
            DeviceKind::IntegratedGpu
        } else {
            DeviceKind::DiscreteGpu
        }
    } else if dev_type & CL_DEVICE_TYPE_CPU != 0 {
        DeviceKind::Cpu
    } else {
        DeviceKind::Other // accélérateurs (FPGA…) et types inconnus
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_de_peripherique() {
        assert_eq!(kind_of(CL_DEVICE_TYPE_GPU, true), DeviceKind::IntegratedGpu);
        assert_eq!(kind_of(CL_DEVICE_TYPE_GPU, false), DeviceKind::DiscreteGpu);
        assert_eq!(kind_of(CL_DEVICE_TYPE_CPU, true), DeviceKind::Cpu);
    }

    #[test]
    fn la_decouverte_ne_plante_jamais() {
        // Disponible ou non selon la machine : seul compte l'absence de panique.
        let _ = OpenClBackend.devices();
    }
}
