//! Kernels et résultats côté Python : `_core.compile_kernel`, `KernelIR.launch`, `DeviceArray`.
//!
//! Les tableaux NumPy sont passés par adresse (`arr.ctypes.data`) : le package Python vérifie le
//! type, la contiguïté et la taille, et garde les tableaux en vie (champ `keepalive`) tant que le
//! calcul peut encore les lire.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use gpusync_core::{
    Compiled, CoreError, Device, DeviceArray, Input, Mode, TransferLimits, compile, launch, parse,
    set_transfer_limits, source, transfer_limits,
};
use gpusync_ir::{Const, Grid, Kernel, ParamKind, Scalar};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::{create_exception, types::PyList};

create_exception!(
    gpusync,
    GpusyncError,
    PyException,
    "Erreur de base de GPUsync."
);
create_exception!(
    gpusync,
    KernelSyntaxError,
    GpusyncError,
    "Construction Python non supportée dans un kernel."
);
create_exception!(
    gpusync,
    TypeMismatch,
    GpusyncError,
    "Type d'argument ou d'expression incompatible."
);
create_exception!(
    gpusync,
    ShapeMismatch,
    GpusyncError,
    "Formes de tableaux incompatibles."
);
create_exception!(
    gpusync,
    UnsupportedFeature,
    GpusyncError,
    "Capacité absente du périphérique (ex. f64)."
);
create_exception!(
    gpusync,
    DeviceError,
    GpusyncError,
    "Erreur du driver ou du périphérique."
);

fn to_py(e: CoreError) -> PyErr {
    match e {
        CoreError::Syntax(m) => KernelSyntaxError::new_err(m),
        CoreError::Type(m) => TypeMismatch::new_err(m),
        CoreError::Shape(m) => ShapeMismatch::new_err(m),
        CoreError::Unsupported(m) => UnsupportedFeature::new_err(m),
        CoreError::Device(m) => DeviceError::new_err(m),
    }
}

/// Kernel analysé (IR), avec ses versions compilées par périphérique.
#[pyclass(module = "gpusync._core", frozen)]
pub struct KernelIR {
    ir: Arc<Kernel>,
    compiled: Mutex<HashMap<String, (Device, Compiled)>>,
}

impl KernelIR {
    /// Version compilée pour `dev` (cache par identifiant, invalidé si le périphérique a été rouvert).
    fn compiled_for(&self, dev: &Device) -> Result<Compiled, CoreError> {
        let id = dev.id();
        if let Some((d, c)) = self
            .compiled
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&id)
        {
            if d.same(dev) {
                return Ok(c.clone());
            }
        }
        let c = compile(dev, &self.ir)?;
        self.compiled
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, (dev.clone(), c.clone()));
        Ok(c)
    }
}

#[pymethods]
impl KernelIR {
    #[getter]
    fn name(&self) -> String {
        self.ir.name.clone()
    }

    /// Grille : "elementwise" (une case de sortie par élément d'entrée) ou "indexed" (`gs.index()`).
    #[getter]
    fn grid(&self) -> &'static str {
        match self.ir.grid {
            Grid::Elementwise => "elementwise",
            Grid::Indexed { .. } => "indexed",
        }
    }

    /// Noms des dimensions passées en paramètres cachés (kernel avec index), dans l'ordre des
    /// scalaires à ajouter après les scalaires visibles.
    #[getter]
    fn dims(&self) -> Vec<String> {
        self.ir
            .params
            .iter()
            .filter_map(|p| p.name.strip_prefix("gs_dim_").filter(|_| p.hidden))
            .map(str::to_string)
            .collect()
    }

    /// Nombre de sorties.
    #[getter]
    fn n_outputs(&self) -> usize {
        self.ir.outputs.len()
    }

    /// Code généré : "opencl", "wgsl" ou "ir".
    fn source(&self, target: &str) -> PyResult<String> {
        source(&self.ir, target).map_err(to_py)
    }

    /// Besoins du kernel : {"f64": bool, "f64_math": bool, "f64_math_op": str | None}.
    #[getter]
    fn requirements(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let r = self.ir.requirements();
        let d = pyo3::types::PyDict::new(py);
        d.set_item("f64", r.f64)?;
        d.set_item("f64_math", r.f64_math)?;
        d.set_item("f64_math_op", r.f64_math_op)?;
        d.set_item("int64", r.int64)?;
        Ok(d.unbind())
    }

    /// Compile pour `device_id` sans lancer (pour mesurer ou vérifier tôt).
    fn prepare(&self, py: Python<'_>, device_id: &str) -> PyResult<()> {
        py.detach(|| {
            let dev = Device::open(device_id)?;
            self.compiled_for(&dev).map(|_| ())
        })
        .map_err(to_py)
    }

    /// Lance le kernel sur une grille de `n` éléments (taille de chaque sortie).
    ///
    /// - `arrays` : un élément par paramètre tableau, `(adresse, octets)` d'un tableau NumPy
    ///   contigu, ou un `DeviceArray` ;
    /// - `scalars` : un nombre Python par paramètre scalaire (dimensions cachées comprises, en
    ///   dernier) ;
    /// - `out_ptrs` : adresse d'un tableau NumPy par sortie (utilisé directement en zéro-copie) ;
    /// - `keepalive` : objets à garder en vie jusqu'à la fin du calcul.
    #[pyo3(signature = (device_id, arrays, scalars, n, out_ptrs, keepalive))]
    #[allow(clippy::too_many_arguments)] // signature imposée par l'appel Python
    fn launch(
        &self,
        py: Python<'_>,
        device_id: &str,
        arrays: &Bound<'_, PyList>,
        scalars: &Bound<'_, PyList>,
        n: usize,
        out_ptrs: Vec<usize>,
        keepalive: Vec<Py<PyAny>>,
    ) -> PyResult<Vec<DeviceArrayPy>> {
        // Scalaires : convertis et vérifiés selon le type déclaré.
        let scalar_params: Vec<_> = self
            .ir
            .params
            .iter()
            .enumerate()
            .filter(|(i, p)| p.kind == ParamKind::Scalar && !self.ir.is_engine_param(*i))
            .map(|(_, p)| p)
            .collect();
        if scalars.len() != scalar_params.len() {
            return Err(TypeMismatch::new_err(format!(
                "{} scalaire(s) attendu(s)",
                scalar_params.len()
            )));
        }
        let mut consts = Vec::with_capacity(scalars.len());
        for (v, p) in scalars.iter().zip(&scalar_params) {
            consts.push(match p.ty {
                Scalar::F32 | Scalar::F64 => Const::F64(v.extract::<f64>().map_err(|_| {
                    TypeMismatch::new_err(format!(
                        "paramètre '{}' : nombre attendu, reçu {}",
                        p.name,
                        v.get_type()
                            .name()
                            .map(|n| n.to_string())
                            .unwrap_or_default()
                    ))
                })?),
                Scalar::I32 | Scalar::U32 | Scalar::I64 | Scalar::U64 => {
                    let i: i128 = v.extract().map_err(|_| {
                        TypeMismatch::new_err(format!(
                            "paramètre '{}' ({}) : entier attendu, reçu {}",
                            p.name,
                            p.ty,
                            v.repr().map(|r| r.to_string()).unwrap_or_default()
                        ))
                    })?;
                    let range = || {
                        TypeMismatch::new_err(format!(
                            "paramètre '{}' : {i} hors de la plage de {}",
                            p.name, p.ty
                        ))
                    };
                    match p.ty {
                        Scalar::I32 => Const::I32(i32::try_from(i).map_err(|_| range())?),
                        Scalar::U32 => Const::U32(u32::try_from(i).map_err(|_| range())?),
                        Scalar::I64 => Const::I64(i64::try_from(i).map_err(|_| range())?),
                        _ => Const::U64(u64::try_from(i).map_err(|_| range())?),
                    }
                }
                Scalar::Bool => Const::Bool(v.is_truthy()?),
            });
        }
        // Tableaux : adresse hôte ou résultat d'un kernel précédent.
        enum A {
            Host(usize, usize),
            Dev(Py<DeviceArrayPy>),
        }
        let mut items = Vec::with_capacity(arrays.len());
        for a in arrays.iter() {
            if let Ok(d) = a.cast::<DeviceArrayPy>() {
                items.push(A::Dev(d.clone().unbind()));
            } else {
                let (ptr, bytes): (usize, usize) = a.extract()?;
                items.push(A::Host(ptr, bytes));
            }
        }
        let borrowed: Vec<_> = items
            .iter()
            .map(|i| match i {
                A::Dev(d) => Some(d.get()),
                A::Host(..) => None,
            })
            .collect();
        let result = py.detach(|| -> Result<Vec<DeviceArray>, CoreError> {
            let dev = Device::open(device_id)?;
            let c = self.compiled_for(&dev)?;
            let inputs: Vec<Input> = items
                .iter()
                .zip(&borrowed)
                .map(|(i, b)| match (i, b) {
                    (A::Host(p, n), _) => Input::Host {
                        ptr: *p as *const u8,
                        bytes: *n,
                    },
                    (A::Dev(_), Some(d)) => Input::Device(&d.array),
                    _ => unreachable!(),
                })
                .collect();
            let outs: Vec<*mut u8> = out_ptrs.iter().map(|p| *p as *mut u8).collect();
            unsafe { launch(&dev, &c, &inputs, &consts, n, &outs) }
        });
        let arrays = result.map_err(to_py)?;
        // Entrées issues d'un calcul précédent : gardées en vie seulement si elles s'appuient sur la
        // mémoire de l'hôte (zéro-copie, CPU, flux). Un résultat resté sur le périphérique peut être
        // libéré dès le lancement mis en file : son buffer n'est réutilisé qu'après les commandes qui
        // le lisent (ordre des files, événements du pool). Le garder bloquerait toute réutilisation
        // dans une boucle `y = f(y)` (chaque appel allouerait de la mémoire neuve).
        let mut keep = keepalive;
        for i in items {
            if let A::Dev(d) = i {
                if d.get().array.mode != Mode::Resident {
                    keep.push(d.into_any());
                }
            }
        }
        // Chaque sortie garde ses propres références : les entrées vivent tant qu'une sortie
        // peut encore être calculée.
        Ok(arrays
            .into_iter()
            .map(|array| DeviceArrayPy {
                array,
                keepalive: Mutex::new(keep.iter().map(|o| o.clone_ref(py)).collect()),
            })
            .collect())
    }
}

/// Résultat d'un kernel, resté sur le périphérique jusqu'à `read_into`.
#[pyclass(name = "DeviceArray", module = "gpusync._core", frozen)]
pub struct DeviceArrayPy {
    // Ordre important : `array` est détruit en premier (il attend la fin des calculs), puis
    // seulement les tableaux NumPy qu'il pouvait encore lire.
    array: DeviceArray,
    keepalive: Mutex<Vec<Py<PyAny>>>,
}

#[pymethods]
impl DeviceArrayPy {
    #[getter]
    fn device_id(&self) -> String {
        self.array.device.id()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.array.bytes()
    }

    /// Mode d'exécution : "direct" (zéro-copie), "resident", "flux" ou "cpu".
    #[getter]
    fn mode(&self) -> &'static str {
        self.array.mode.name()
    }

    /// Adresse hôte du résultat s'il est en zéro-copie (sinon None).
    #[getter]
    fn host_ptr(&self) -> Option<usize> {
        self.array.host_ptr().map(|p| p as usize)
    }

    /// Attend la fin du calcul et copie le résultat à l'adresse `ptr` (sans copie si c'est déjà
    /// la mémoire du résultat). Libère ensuite les entrées gardées en vie.
    fn read_into(&self, py: Python<'_>, ptr: usize) -> PyResult<()> {
        py.detach(|| unsafe { self.array.read_into(ptr as *mut u8) })
            .map_err(to_py)?;
        self.keepalive
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        Ok(())
    }
}

/// Analyse un kernel (JSON produit par `gpusync._ast`) et renvoie son IR.
#[pyfunction]
fn compile_kernel(spec_json: &str) -> PyResult<KernelIR> {
    let ir = parse(spec_json).map_err(to_py)?;
    Ok(KernelIR {
        ir,
        compiled: Mutex::new(HashMap::new()),
    })
}

/// Réglages du moteur de transferts : {"chunk_bytes", "budget_bytes", "direct_chunk_bytes"}.
#[pyfunction]
fn get_transfer_limits(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
    let l = transfer_limits();
    let d = pyo3::types::PyDict::new(py);
    d.set_item("chunk_bytes", l.chunk_bytes)?;
    d.set_item("budget_bytes", l.budget_bytes)?;
    d.set_item("direct_chunk_bytes", l.direct_chunk_bytes)?;
    Ok(d.unbind())
}

/// Modifie les réglages du moteur de transferts (None : valeur par défaut pour budget et bloc Direct).
#[pyfunction]
#[pyo3(name = "set_transfer_limits", signature = (chunk_bytes, budget_bytes, direct_chunk_bytes))]
fn set_transfer_limits_py(
    chunk_bytes: usize,
    budget_bytes: Option<u64>,
    direct_chunk_bytes: Option<usize>,
) -> PyResult<()> {
    if chunk_bytes < 64 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "chunk_bytes : au moins 64 octets",
        ));
    }
    set_transfer_limits(TransferLimits {
        chunk_bytes,
        budget_bytes,
        direct_chunk_bytes,
    });
    Ok(())
}

/// Le périphérique travaille-t-il directement dans la mémoire de l'hôte ?
#[pyfunction]
fn device_zero_copy(py: Python<'_>, device_id: &str) -> PyResult<bool> {
    py.detach(|| Device::open(device_id).map(|d| d.zero_copy()))
        .map_err(to_py)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("GpusyncError", py.get_type::<GpusyncError>())?;
    m.add("KernelSyntaxError", py.get_type::<KernelSyntaxError>())?;
    m.add("TypeMismatch", py.get_type::<TypeMismatch>())?;
    m.add("ShapeMismatch", py.get_type::<ShapeMismatch>())?;
    m.add("UnsupportedFeature", py.get_type::<UnsupportedFeature>())?;
    m.add("DeviceError", py.get_type::<DeviceError>())?;
    m.add_class::<KernelIR>()?;
    m.add_class::<DeviceArrayPy>()?;
    m.add_function(wrap_pyfunction!(compile_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(device_zero_copy, m)?)?;
    m.add_function(wrap_pyfunction!(get_transfer_limits, m)?)?;
    m.add_function(wrap_pyfunction!(set_transfer_limits_py, m)?)?;
    Ok(())
}
