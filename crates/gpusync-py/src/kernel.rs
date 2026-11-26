//! Kernels and results on the Python side: `_core.compile_kernel`, `KernelIR.launch`, `DeviceArray`.
//!
//! NumPy arrays are passed by address (`arr.ctypes.data`): the Python package checks the
//! type, contiguity and size, and keeps the arrays alive (`keepalive` field) as long as the
//! computation may still read them.

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
    "Base error of GPUsync."
);
create_exception!(
    gpusync,
    KernelSyntaxError,
    GpusyncError,
    "Python construct not supported in a kernel."
);
create_exception!(
    gpusync,
    TypeMismatch,
    GpusyncError,
    "Incompatible argument or expression type."
);
create_exception!(
    gpusync,
    ShapeMismatch,
    GpusyncError,
    "Incompatible array shapes."
);
create_exception!(
    gpusync,
    UnsupportedFeature,
    GpusyncError,
    "Capability missing on the device (e.g. f64)."
);
create_exception!(
    gpusync,
    DeviceError,
    GpusyncError,
    "Driver or device error."
);

pub(crate) fn to_py(e: CoreError) -> PyErr {
    match e {
        CoreError::Syntax(m) => KernelSyntaxError::new_err(m),
        CoreError::Type(m) => TypeMismatch::new_err(m),
        CoreError::Shape(m) => ShapeMismatch::new_err(m),
        CoreError::Unsupported(m) => UnsupportedFeature::new_err(m),
        CoreError::Device(m) => DeviceError::new_err(m),
    }
}

/// Analysed kernel (IR), with its compiled versions per device.
#[pyclass(module = "gpusync._core", frozen)]
pub struct KernelIR {
    ir: Arc<Kernel>,
    compiled: Mutex<HashMap<String, (Device, Compiled)>>,
}

impl KernelIR {
    /// Compiled version for `dev` (cached by identifier, invalidated if the device was reopened).
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

    /// Grid: "elementwise" (one output cell per input element) or "indexed" (`gs.index()`).
    #[getter]
    fn grid(&self) -> &'static str {
        match self.ir.grid {
            Grid::Elementwise => "elementwise",
            Grid::Indexed { .. } => "indexed",
        }
    }

    /// Names of the dimensions passed as hidden parameters (indexed kernel), in the order of the
    /// scalars to append after the visible ones.
    #[getter]
    fn dims(&self) -> Vec<String> {
        self.ir
            .params
            .iter()
            .filter_map(|p| p.name.strip_prefix("gs_dim_").filter(|_| p.hidden))
            .map(str::to_string)
            .collect()
    }

    /// Number of outputs.
    #[getter]
    fn n_outputs(&self) -> usize {
        self.ir.outputs.len()
    }

    /// Generated code: "opencl", "wgsl" or "ir".
    fn source(&self, target: &str) -> PyResult<String> {
        source(&self.ir, target).map_err(to_py)
    }

    /// Requirements of the kernel: {"f64": bool, "f64_math": bool, "f64_math_op": str | None}.
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

    /// Compiles for `device_id` without launching (to measure or check early).
    fn prepare(&self, py: Python<'_>, device_id: &str) -> PyResult<()> {
        py.detach(|| {
            let dev = Device::open(device_id)?;
            self.compiled_for(&dev).map(|_| ())
        })
        .map_err(to_py)
    }

    /// Launches the kernel over a grid of `n` elements (size of each output).
    ///
    /// - `arrays`: one item per array parameter, `(address, bytes)` of a contiguous NumPy
    ///   array, or a `DeviceArray`;
    /// - `scalars`: one Python number per scalar parameter (hidden dimensions included,
    ///   last);
    /// - `out_ptrs`: address of a NumPy array per output (used directly in zero-copy);
    /// - `keepalive`: objects to keep alive until the computation ends.
    #[pyo3(signature = (device_id, arrays, scalars, n, out_ptrs, keepalive))]
    #[allow(clippy::too_many_arguments)] // signature imposed by the Python call
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
        // Scalars: converted and checked against the declared type.
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
                "{} scalar(s) expected",
                scalar_params.len()
            )));
        }
        let mut consts = Vec::with_capacity(scalars.len());
        for (v, p) in scalars.iter().zip(&scalar_params) {
            consts.push(match p.ty {
                Scalar::F32 | Scalar::F64 => Const::F64(v.extract::<f64>().map_err(|_| {
                    TypeMismatch::new_err(format!(
                        "parameter '{}': number expected, got {}",
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
                            "parameter '{}' ({}): integer expected, got {}",
                            p.name,
                            p.ty,
                            v.repr().map(|r| r.to_string()).unwrap_or_default()
                        ))
                    })?;
                    let range = || {
                        TypeMismatch::new_err(format!(
                            "parameter '{}': {i} out of the range of {}",
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
        // Arrays: host address or result of a previous kernel.
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
        // Inputs from a previous computation: kept alive only if they rely on host
        // memory (zero-copy, CPU, flux). A result kept on the device can be released
        // as soon as the launch is queued: its buffer is reused only after the commands that
        // read it (queue order, pool events). Keeping it would block any reuse
        // in a `y = f(y)` loop (each call would allocate fresh memory).
        let mut keep = keepalive;
        for i in items {
            if let A::Dev(d) = i {
                if d.get().array.mode != Mode::Resident {
                    keep.push(d.into_any());
                }
            }
        }
        // Each output keeps its own references: inputs live as long as an output
        // may still be computed.
        Ok(arrays
            .into_iter()
            .map(|array| DeviceArrayPy {
                array,
                keepalive: Mutex::new(keep.iter().map(|o| o.clone_ref(py)).collect()),
            })
            .collect())
    }
}

/// Result of a kernel, kept on the device until `read_into`.
#[pyclass(name = "DeviceArray", module = "gpusync._core", frozen)]
pub struct DeviceArrayPy {
    // Order matters: `array` is dropped first (it waits for the computations), and only
    // then the NumPy arrays it could still read.
    pub(crate) array: DeviceArray,
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

    /// Execution mode: "direct" (zero-copy), "resident", "flux" or "cpu".
    #[getter]
    fn mode(&self) -> &'static str {
        self.array.mode.name()
    }

    /// Host address of the result if it is zero-copy (otherwise None).
    #[getter]
    fn host_ptr(&self) -> Option<usize> {
        self.array.host_ptr().map(|p| p as usize)
    }

    /// Waits for the computation and copies the result to address `ptr` (no copy if it already is
    /// the result's memory). Then releases the inputs kept alive.
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

/// Analyses a kernel (JSON produced by `gpusync._ast`) and returns its IR.
#[pyfunction]
fn compile_kernel(spec_json: &str) -> PyResult<KernelIR> {
    let ir = parse(spec_json).map_err(to_py)?;
    Ok(KernelIR {
        ir,
        compiled: Mutex::new(HashMap::new()),
    })
}

/// Settings of the transfer engine: {"chunk_bytes", "budget_bytes", "direct_chunk_bytes"}.
#[pyfunction]
fn get_transfer_limits(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
    let l = transfer_limits();
    let d = pyo3::types::PyDict::new(py);
    d.set_item("chunk_bytes", l.chunk_bytes)?;
    d.set_item("budget_bytes", l.budget_bytes)?;
    d.set_item("direct_chunk_bytes", l.direct_chunk_bytes)?;
    Ok(d.unbind())
}

/// Changes the transfer engine settings (None: default value for the budget and Direct chunk).
#[pyfunction]
#[pyo3(name = "set_transfer_limits", signature = (chunk_bytes, budget_bytes, direct_chunk_bytes))]
fn set_transfer_limits_py(
    chunk_bytes: usize,
    budget_bytes: Option<u64>,
    direct_chunk_bytes: Option<usize>,
) -> PyResult<()> {
    if chunk_bytes < 64 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "chunk_bytes: at least 64 bytes",
        ));
    }
    set_transfer_limits(TransferLimits {
        chunk_bytes,
        budget_bytes,
        direct_chunk_bytes,
    });
    Ok(())
}

/// Does the device work directly in host memory?
#[pyfunction]
fn device_zero_copy(py: Python<'_>, device_id: &str) -> PyResult<bool> {
    py.detach(|| Device::open(device_id).map(|d| d.zero_copy()))
        .map_err(to_py)
}

/// Opens a command batch on `device_id` (wgpu: a single CommandEncoder for several kernels).
#[pyfunction]
fn begin_batch(py: Python<'_>, device_id: &str) -> PyResult<()> {
    py.detach(|| {
        let d = Device::open(device_id)?;
        d.begin_batch();
        Ok(())
    })
    .map_err(to_py)
}

/// Closes the current command batch and submits the work to the GPU.
#[pyfunction]
fn end_batch(py: Python<'_>, device_id: &str) -> PyResult<()> {
    py.detach(|| Device::open(device_id)?.end_batch())
        .map_err(to_py)
}

/// Waits for all work submitted on `device_id` (without copying data back to RAM).
#[pyfunction]
fn sync(py: Python<'_>, device_id: &str) -> PyResult<()> {
    py.detach(|| Device::open(device_id)?.finish())
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
    m.add_function(wrap_pyfunction!(begin_batch, m)?)?;
    m.add_function(wrap_pyfunction!(end_batch, m)?)?;
    m.add_function(wrap_pyfunction!(sync, m)?)?;
    m.add_function(wrap_pyfunction!(get_transfer_limits, m)?)?;
    m.add_function(wrap_pyfunction!(set_transfer_limits_py, m)?)?;
    Ok(())
}
