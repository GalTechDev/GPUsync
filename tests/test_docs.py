"""The examples of docs/markdown work: every ```python block is run, page by page, in order.

Blocks of a page share their variables (like a notebook). They are written to a file, because a
kernel needs readable source code. ```py blocks (window, PyTorch, large files) are not run.
"""

import re
import runpy
import warnings
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs" / "markdown"
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def _pages():
    return sorted(p for p in DOCS.glob("*.md") if BLOCK.search(p.read_text(encoding="utf-8")))


@pytest.mark.parametrize("page", _pages(), ids=lambda p: p.name)
def test_examples_of_the_page_run(page, tmp_path, monkeypatch, capsys):
    blocks = BLOCK.findall(page.read_text(encoding="utf-8"))
    # One file per page: error line numbers point to the faulty block
    script = tmp_path / f"doc_{page.stem}.py"
    parts = [f"# --- block {k + 1}\n{b}" for k, b in enumerate(blocks)]
    script.write_text("\n".join(parts), encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # examples that write files do it here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # device fallbacks announced by the examples themselves
        runpy.run_path(str(script), run_name="__main__")


def test_site_pages_are_up_to_date():
    """The site (docs/*.html, GitHub Pages) is generated from docs/*.md: after changing the docs, run
    `python3 tools/build_pages.py` again."""
    import subprocess
    import sys

    root = DOCS.parent.parent
    r = subprocess.run([sys.executable, str(root / "tools" / "build_pages.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
