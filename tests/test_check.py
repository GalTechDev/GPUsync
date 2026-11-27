"""`python -m gpusync --check`: the self-test passes on every device and reports failures."""

import pytest

import gpusync as gs
from gpusync import __main__ as cli
from gpusync import _check


def test_every_device_passes():
    results = _check.check()
    assert {r.device.id for r in results} == {d.id for d in gs.devices()}
    for r in results:
        assert set(r.status) == {name for name, _ in _check.TESTS}
        assert r.ok, f"{r.device.id}: {r.errors}"
        assert (r.status["f64"] == "ok") == r.device.f64


def test_command(capsys):
    assert cli.main(["--check", "-d", "cpu:0"]) == 0
    out = capsys.readouterr().out
    assert "cpu:0" in out and "✓ atomics" in out and "All 1 device(s) passed." in out


def test_failure_is_reported(monkeypatch, capsys):
    def wrong(dev, rng):
        import numpy as np

        np.testing.assert_allclose(np.ones(4), np.zeros(4))

    monkeypatch.setattr(_check, "TESTS", _check.TESTS[:1] + [("broken", wrong)])
    assert cli.main(["--check", "-d", "cpu:0"]) == 1
    out = capsys.readouterr().out
    assert "✗ broken" in out and "broken: wrong result: Mismatched elements: 4 / 4" in out
    assert "FAILED on cpu:0." in out


def test_unknown_device(capsys):
    assert cli.main(["--check", "-d", "bogus:3"]) == 2
    assert "unknown device(s): bogus:3" in capsys.readouterr().out


def test_device_needs_check():
    with pytest.raises(SystemExit):
        cli.main(["-d", "cpu:0"])
