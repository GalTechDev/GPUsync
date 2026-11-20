//! Runtime GPUsync : périphériques, backends et (plus tard) buffers, streams et ordonnancement.
//!
//! M0 : découverte des périphériques. Chaque backend implémente [`Backend`] ; le runtime agrège
//! leurs périphériques et retient l'état de chaque backend (disponible ou non, et pourquoi).
//! Voir `ARCHITECTURE.md` §8.

use std::fmt;

/// Catégorie de périphérique, telle que rapportée par le driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    Cpu,
    Other,
}

impl DeviceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceKind::DiscreteGpu => "discrete_gpu",
            DeviceKind::IntegratedGpu => "integrated_gpu",
            DeviceKind::VirtualGpu => "virtual_gpu",
            DeviceKind::Cpu => "cpu",
            DeviceKind::Other => "other",
        }
    }

    /// Rang pour un ordre stable : GPU dédiés, intégrés, virtuels, CPU, autres.
    pub fn rank(self) -> u8 {
        match self {
            DeviceKind::DiscreteGpu => 0,
            DeviceKind::IntegratedGpu => 1,
            DeviceKind::VirtualGpu => 2,
            DeviceKind::Cpu => 3,
            DeviceKind::Other => 4,
        }
    }

    pub fn is_gpu(self) -> bool {
        matches!(
            self,
            DeviceKind::DiscreteGpu | DeviceKind::IntegratedGpu | DeviceKind::VirtualGpu
        )
    }
}

/// Description d'un périphérique de calcul découvert par un backend.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceInfo {
    /// Nom commercial, ex. « NVIDIA GeForce RTX 4080 SUPER ».
    pub name: String,
    /// API utilisée : "metal", "vulkan", "dx12" (via wgpu) ou "opencl".
    pub backend: &'static str,
    /// Index parmi les périphériques du même backend (`"opencl:1"`).
    pub index: usize,
    pub kind: DeviceKind,
    /// Fabricant tel que rapporté par le driver.
    pub vendor: String,
    /// Version de l'API ou du driver, ex. « OpenCL 3.0 CUDA » ou « 610.88 ».
    pub version: String,
    /// Le périphérique sait-il calculer en float64 (arithmétique) ?
    pub f64: bool,
    /// … y compris les fonctions transcendantes (exp, log, sin…) en f64 ?
    pub f64_math: bool,
    /// Mémoire partagée avec l'hôte (zéro-copie possible, §8.4 de l'architecture).
    pub unified_memory: bool,
    /// Mémoire globale en octets, si le backend l'expose (wgpu ne l'expose pas).
    pub memory_bytes: Option<u64>,
    /// Taille max d'un buffer, en octets (contraint le découpage out-of-core).
    pub max_buffer_bytes: Option<u64>,
}

impl DeviceInfo {
    /// Identifiant court : `"opencl:0"`, `"metal:0"`…
    pub fn id(&self) -> String {
        format!("{}:{}", self.backend, self.index)
    }
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.id())
    }
}

/// Erreur d'un backend (bibliothèque absente, driver en échec…).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendError(pub String);

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BackendError {}

/// Un backend de calcul (wgpu, OpenCL, CUDA…).
///
/// M0 ne demande que la découverte. Les opérations (allocation, lancement, attente) viendront
/// avec M1, selon le trait décrit au §8.2 de l'architecture.
pub trait Backend: Send + Sync {
    /// Nom du backend : "wgpu", "opencl"…
    fn name(&self) -> &'static str;

    /// Liste les périphériques. Une erreur signifie « backend indisponible sur cette machine » :
    /// elle ne doit jamais faire planter l'appelant.
    fn devices(&self) -> Result<Vec<DeviceInfo>, BackendError>;
}

/// État d'un backend après découverte.
#[derive(Debug, Clone)]
pub struct BackendStatus {
    pub name: &'static str,
    pub devices: Vec<DeviceInfo>,
    /// `None` si le backend est disponible ; sinon la raison.
    pub error: Option<String>,
}

impl BackendStatus {
    pub fn available(&self) -> bool {
        self.error.is_none()
    }
}

/// Interroge chaque backend. Un backend en erreur est signalé, jamais fatal (objectif O1).
pub fn discover(backends: &[&dyn Backend]) -> Vec<BackendStatus> {
    backends
        .iter()
        .map(|b| {
            // Un driver défaillant peut paniquer dans du code tiers : on l'isole aussi.
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| b.devices())) {
                Ok(Ok(devices)) => BackendStatus {
                    name: b.name(),
                    devices,
                    error: None,
                },
                Ok(Err(e)) => BackendStatus {
                    name: b.name(),
                    devices: vec![],
                    error: Some(e.0),
                },
                Err(_) => BackendStatus {
                    name: b.name(),
                    devices: vec![],
                    error: Some("le backend a paniqué pendant la découverte".into()),
                },
            }
        })
        .collect()
}

/// Liste à plat de tous les périphériques, dans l'ordre de préférence des backends fournis.
pub fn all_devices(statuses: &[BackendStatus]) -> Vec<DeviceInfo> {
    statuses
        .iter()
        .flat_map(|s| s.devices.iter().cloned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(&'static str, Result<usize, &'static str>);

    impl Backend for Fake {
        fn name(&self) -> &'static str {
            self.0
        }
        fn devices(&self) -> Result<Vec<DeviceInfo>, BackendError> {
            match self.1 {
                Ok(n) => Ok((0..n)
                    .map(|i| DeviceInfo {
                        name: format!("GPU {i}"),
                        backend: self.0,
                        index: i,
                        kind: DeviceKind::DiscreteGpu,
                        vendor: "test".into(),
                        version: "1".into(),
                        f64: true,
                        f64_math: true,
                        unified_memory: false,
                        memory_bytes: None,
                        max_buffer_bytes: None,
                    })
                    .collect()),
                Err(e) => Err(BackendError(e.into())),
            }
        }
    }

    struct Panics;

    impl Backend for Panics {
        fn name(&self) -> &'static str {
            "panique"
        }
        fn devices(&self) -> Result<Vec<DeviceInfo>, BackendError> {
            panic!("driver cassé")
        }
    }

    #[test]
    fn un_backend_en_erreur_ou_qui_panique_n_empeche_pas_les_autres() {
        let ok = Fake("ok", Ok(2));
        let ko = Fake("ko", Err("bibliothèque introuvable"));
        let st = discover(&[&ko, &Panics, &ok]);
        assert_eq!(st.len(), 3);
        assert_eq!(st[0].error.as_deref(), Some("bibliothèque introuvable"));
        assert!(st[1].error.as_deref().unwrap().contains("paniqué"));
        assert!(st[2].available());
        let devs = all_devices(&st);
        assert_eq!(devs.len(), 2);
        assert_eq!(devs[1].id(), "ok:1");
    }
}
