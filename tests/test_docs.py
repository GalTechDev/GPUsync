"""Les exemples de docs/ marchent : chaque bloc ```python est exécuté, page par page, dans l'ordre.

Les blocs d'une page partagent leurs variables (comme un notebook). Ils sont écrits dans un fichier, car un
kernel doit avoir un code source lisible. Les blocs ```py (fenêtre, PyTorch, gros fichiers) ne sont pas lancés.
"""

import re
import runpy
import warnings
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def _pages():
    return sorted(p for p in DOCS.glob("*.md") if BLOCK.search(p.read_text(encoding="utf-8")))


@pytest.mark.parametrize("page", _pages(), ids=lambda p: p.name)
def test_examples_of_the_page_run(page, tmp_path, monkeypatch, capsys):
    blocks = BLOCK.findall(page.read_text(encoding="utf-8"))
    # Un fichier par page : les numéros de ligne des erreurs renvoient au bloc fautif
    script = tmp_path / f"doc_{page.stem}.py"
    parts = [f"# --- bloc {k + 1}\n{b}" for k, b in enumerate(blocks)]
    script.write_text("\n".join(parts), encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # les exemples qui écrivent des fichiers le font ici
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # replis de périphérique annoncés par les exemples eux-mêmes
        runpy.run_path(str(script), run_name="__main__")


def test_site_pages_are_up_to_date():
    """docs/pages (GitHub Pages) est généré depuis docs/*.md : après une modification de la doc, relancer
    `python3 tools/build_pages.py`."""
    import subprocess
    import sys

    root = DOCS.parent
    r = subprocess.run([sys.executable, str(root / "tools" / "build_pages.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
