"""Tests du moteur gpusync-gfx : ignorés (et signalés) s'il n'est pas installé, pour que la bibliothèque seule se
teste sans lui."""

import importlib.util

if importlib.util.find_spec("gpusync.gfx") is None:
    collect_ignore_glob = ["test_*.py"]

    def pytest_report_header(config):
        return "gpusync-gfx non installé : tests du moteur ignorés (pip install -e gfx)"
