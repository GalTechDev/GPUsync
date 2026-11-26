"""Tests of the gpusync-gfx engine: skipped (and reported) if it is not installed, so that the library
alone can be tested without it."""

import importlib.util

if importlib.util.find_spec("gpusync.gfx") is None:
    collect_ignore_glob = ["test_*.py"]

    def pytest_report_header(config):
        return "gpusync-gfx not installed: engine tests skipped (pip install -e gfx)"
