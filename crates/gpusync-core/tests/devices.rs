//! Differential test: the `sample` kernel (real Python AST, tests/fixtures/sample.json) runs
//! on every available device and is compared with the reference interpreter.

use gpusync_core::{Device, Input, compile, launch, parse};
use gpusync_ir::{Const, Scalar};
use gpusync_runtime::Backend;

const SAMPLE: &str = include_str!("fixtures/sample.json");

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

fn all_device_ids() -> Vec<String> {
    let mut ids = vec!["cpu:0".to_string()];
    for b in [
        &gpusync_backend_wgpu::WgpuBackend as &dyn Backend,
        &gpusync_backend_opencl::OpenClBackend,
    ] {
        if let Ok(devs) = b.devices() {
            ids.extend(devs.iter().map(|d| d.id()));
        }
    }
    ids
}

#[test]
fn frontend_local_types() {
    let k = parse(SAMPLE).unwrap();
    let ty = |n: &str| k.locals.iter().find(|l| l.name == n).unwrap().ty;
    assert_eq!(ty("acc"), Scalar::F32, "acc = 0 then acc += x*m → f32");
    assert_eq!(ty("i"), Scalar::I32);
    assert_eq!(ty("y"), Scalar::F32);
    assert!(!k.requirements().f64, "no implicit promotion to f64");
}

#[test]
fn same_result_on_all_devices() {
    let k = parse(SAMPLE).unwrap();
    let n = 10_007; // not a multiple of the group size
    let x: Vec<f32> = (0..n).map(|i| -5.0 + 10.0 * i as f32 / n as f32).collect();
    let kk: Vec<i32> = (0..n as i32).map(|i| i - n as i32 / 2).collect();
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let kb: Vec<u8> = kk.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let run = |id: &str| -> Vec<f32> {
        let dev = Device::open(id).unwrap();
        let c = compile(&dev, &k).unwrap_or_else(|e| panic!("{id}: {e}"));
        let mut out = vec![0u8; n * 4];
        unsafe {
            let mut arrs = launch(
                &dev,
                &c,
                &[
                    Input::Host {
                        ptr: xb.as_ptr(),
                        bytes: xb.len(),
                    },
                    Input::Host {
                        ptr: kb.as_ptr(),
                        bytes: kb.len(),
                    },
                ],
                &[Const::F32(0.5)],
                n,
                &[out.as_mut_ptr()],
            )
            .unwrap_or_else(|e| panic!("{id}: {e}"));
            let arr = arrs.pop().unwrap();
            if dev.zero_copy() {
                assert_eq!(
                    arr.host_ptr(),
                    Some(out.as_mut_ptr()),
                    "{id}: the output should be zero-copy"
                );
            }
            arr.read_into(out.as_mut_ptr()).unwrap();
        }
        f32s(&out)
    };
    let reference = run("cpu:0");
    for id in all_device_ids().iter().skip(1) {
        let got = run(id);
        let mut worst = 0f32;
        for (i, (a, b)) in got.iter().zip(&reference).enumerate() {
            let tol = 1e-5 * b.abs().max(1.0);
            let d = (a - b).abs();
            worst = worst.max(d / b.abs().max(1.0));
            assert!(
                d <= tol,
                "{id}: element {i} (x = {}, k = {}): {a} instead of {b}",
                x[i],
                kk[i]
            );
        }
        eprintln!("{id:<10} OK — max relative error {worst:.2e}");
    }
}
