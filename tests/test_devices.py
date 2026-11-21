"""Tests de la découverte des périphériques (jalon M0).

Variable optionnelle GS_EXPECT : backends qui DOIVENT avoir au moins un périphérique sur la machine
de test, ex. GS_EXPECT=metal,opencl (Mac) ou GS_EXPECT=vulkan,opencl (PC). Sans elle, seule la
cohérence est vérifiée (utile sur une CI sans GPU).
"""

import os

import pytest

import gpusync as gs

KINDS = {"discrete_gpu", "integrated_gpu", "virtual_gpu", "cpu", "other"}
BACKENDS = {"cuda", "metal", "vulkan", "dx12", "opencl", "cpu"}


def test_devices_returns_consistent_devices():
    devs = gs.devices()
    assert isinstance(devs, list)
    for d in devs:
        assert isinstance(d, gs.Device)
        assert d.name and isinstance(d.name, str)
        assert d.backend in BACKENDS
        assert d.kind in KINDS
        assert isinstance(d.f64, bool) and isinstance(d.unified_memory, bool)
        assert d.id == f"{d.backend}:{d.index}"
        assert d.memory_bytes is None or d.memory_bytes > 0
        assert d.max_buffer_bytes is None or d.max_buffer_bytes > 0
        assert repr(d).startswith(f"Device('{d.name}', backend='{d.backend}'")
    assert len({d.id for d in devs}) == len(devs), "identifiants en double"


def test_backends_reports_each_backend_without_crashing():
    statuses = gs.backends()
    assert [s.name for s in statuses] == ["cuda", "wgpu", "opencl", "cpu"]
    for s in statuses:
        assert s.available == (s.error is None)
        if not s.available:
            assert s.devices == [] and s.error
    flat = [d.id for s in statuses for d in s.devices]
    assert flat == [d.id for d in gs.devices()]


def test_cache_and_refresh():
    first = [d.id for d in gs.devices()]
    assert [d.id for d in gs.devices()] == first
    assert [d.id for d in gs.devices(refresh=True)] == first


def test_objects_are_immutable():
    devs = gs.devices()
    if not devs:
        pytest.skip("aucun périphérique sur cette machine")
    with pytest.raises(AttributeError):
        devs[0].name = "autre"


@pytest.mark.skipif(not os.environ.get("GS_EXPECT"), reason="GS_EXPECT non défini")
def test_expected_devices_on_this_machine():
    found = {d.backend for d in gs.devices()}
    missing = [b for b in os.environ["GS_EXPECT"].split(",") if b and b not in found]
    assert not missing, f"backends attendus mais sans périphérique : {missing} — état : {gs.backends()}"
