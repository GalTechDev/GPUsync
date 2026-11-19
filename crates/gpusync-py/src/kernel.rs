//! Kernels et résultats côté Python : `_core.compile_kernel`, `KernelIR.launch`, `DeviceArray`.
//!
//! Les tableaux NumPy sont passés par adresse (`arr.ctypes.data`) : le package Python vérifie le
//! type, la contiguïté et la taille, et garde les tableaux en vie (champ `keepalive`) tant que le
//! calcul peut encore les lire.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use gpusync_core::{
    Compiled, CoreError, Device, DeviceArray, Input, compile, launch, parse, source,
};
use gpusync_ir::{Const, Kernel, ParamKind, Scalar};
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
    compiled: Mutex<HashMap<String, Compiled>>,
}

impl KernelIR {
    fn compiled_for(&self, dev: &Device) -> Result<Compiled, CoreError> {
        let id = dev.id();
        if let Some(c) = self
            .compiled
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&id)
        {
            return Ok(c.clone());
        }
        let c = compile(dev, &self.ir)?;
        self.compiled
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, c.clone());
        Ok(c)
    }
}

#[pymethods]
impl KernelIR {
    #[getter]
    fn name(&self) -> String {
        self.ir.name.clone()
    }

    /// Code généré : "opencl", "wgsl" ou "ir".
    fn source(&self, target: &str) -> PyResult<String> {
        source(&self.ir, target).map_err(to_py)
    }

    /// Le kernel utilise-t-il f64 ?
    #[getter]
    fn uses_f64(&self) -> bool {
        self.ir.requirements().f64
    }

    /// Compile pour `device_id` sans lancer (pour mesurer ou vérifier tôt).
    fn prepare(&self, py: Python<'_>, device_id: &str) -> PyResult<()> {
        py.detach(|| {
            let dev = Device::open(device_id)?;
            self.compiled_for(&dev).map(|_| ())
        })
        .map_err(to_py)
    }

    /// Lance le kernel sur `n` éléments.
    ///
    /// - `arrays` : un élément par paramètre tableau, `(adresse, octets)` d'un tableau NumPy
    ///   contigu, ou un `DeviceArray` ;
    /// - `scalars` : un nombre Python par paramètre scalaire ;
    /// - `out_ptr` : adresse d'un tableau NumPy de sortie (utilisé directement en zéro-copie) ;
    /// - `keepalive` : objets à garder en vie jusqu'à la fin du calcul.
    #[pyo3(signature = (device_id, arrays, scalars, n, out_ptr, keepalive))]
    #[allow(clippy::too_many_arguments)] // signature imposée par l'appel Python
    fn launch(
        &self,
        py: Python<'_>,
        device_id: &str,
        arrays: &Bound<'_, PyList>,
        scalars: &Bound<'_, PyList>,
        n: usize,
        out_ptr: usize,
        keepalive: Vec<Py<PyAny>>,
    ) -> PyResult<DeviceArrayPy> {
        // Scalaires : convertis et vérifiés selon le type déclaré.
        let scalar_params: Vec<_> = self
            .ir
            .params
            .iter()
            .filter(|p| p.kind == ParamKind::Scalar)
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
                Scalar::I32 | Scalar::U32 => {
                    let i: i64 = v.extract().map_err(|_| {
                        TypeMismatch::new_err(format!(
                            "paramètre '{}' ({}) : entier attendu, reçu {}",
                            p.name,
                            p.ty,
                            v.repr().map(|r| r.to_string()).unwrap_or_default()
                        ))
                    })?;
                    match p.ty {
                        Scalar::I32 => Const::I32(i32::try_from(i).map_err(|_| {
                            TypeMismatch::new_err(format!(
                                "paramètre '{}' : {i} hors de la plage de i32",
                                p.name
                            ))
                        })?),
                        _ => Const::U32(u32::try_from(i).map_err(|_| {
                            TypeMismatch::new_err(format!(
                                "paramètre '{}' : {i} hors de la plage de u32",
                                p.name
                            ))
                        })?),
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
        let result = py.detach(|| -> Result<DeviceArray, CoreError> {
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
            unsafe { launch(&dev, &c, &inputs, &consts, n, Some(out_ptr as *mut u8)) }
        });
        let array = result.map_err(to_py)?;
        // Les DeviceArray d'entrée doivent aussi rester en vie (leurs buffers sont lus).
        let mut keep = keepalive;
        for i in items {
            if let A::Dev(d) = i {
                keep.push(d.into_any());
            }
        }
        Ok(DeviceArrayPy {
            array,
            keepalive: Mutex::new(keep),
        })
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
    Ok(())
}
