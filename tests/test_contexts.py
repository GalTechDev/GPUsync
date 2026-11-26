"""Milestone M4: the same code in a script, in asyncio code and in Jupyter (the milestone's criterion).

Each variant runs in a fresh process; Jupyter is really executed (nbclient + ipykernel,
development dependencies), the kernel being defined in a notebook cell.
"""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

KERNEL = textwrap.dedent('''
    import numpy as np
    import gpusync as gs

    @gs.kernel
    def pairwise(x: gs.f32["n", "t"]) -> gs.f32["n", "n"]:
        i, j = gs.index()
        acc = 0.0
        for k in range(x.shape[1]):
            d = x[i, k] - x[j, k]
            acc += d * d
        return gs.sqrt(acc)

    @gs.kernel
    def stats(x: gs.f32["n", "t"]) -> (gs.f32["n"], gs.f32["n"]):
        i = gs.index()
        s = 0.0
        for k in range(x.shape[1]):
            s += x[i, k]
        return s, s / x.shape[1]

    x = np.random.default_rng(5).random((300, 12), dtype=np.float32)
''')

SYNC = textwrap.dedent('''
    d = pairwise(x)
    s, m = stats(x)
    print("RESULT", json.dumps([float(np.asarray(d).sum()), float(s.sum()), float(m.mean()), d.shape[0]]))
''')

ASYNC = textwrap.dedent('''
    async def main():
        d = await pairwise(x)
        s, m = await stats(x)
        return d, s, m

    d, s, m = asyncio.run(main())
    print("RESULT", json.dumps([float(np.asarray(d).sum()), float(s.sum()), float(m.mean()), d.shape[0]]))
''')

AWAIT_TOP = textwrap.dedent('''
    d = await pairwise(x)
    s, m = await stats(x)
    print("RESULT", json.dumps([float(np.asarray(d).sum()), float(s.sum()), float(m.mean()), d.shape[0]]))
''')


def reference():
    x = np.random.default_rng(5).random((300, 12), dtype=np.float32)
    d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    s = x.sum(1)
    return [float(d.sum()), float(s.sum()), float((s / 12).mean()), 300]


def parse(out: str):
    line = next(l for l in out.splitlines() if l.startswith("RESULT"))
    return json.loads(line.split(" ", 1)[1])


def same(got, ref):
    assert got[3] == ref[3]
    np.testing.assert_allclose(got[:3], ref[:3], rtol=1e-4)


@pytest.mark.parametrize("variant", ["script", "asyncio"])
def test_script_and_asyncio(variant, tmp_path):
    body = SYNC if variant == "script" else ASYNC
    path = tmp_path / f"variante_{variant}.py"
    path.write_text("import asyncio, json\n" + KERNEL + body)
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    same(parse(r.stdout), reference())


@pytest.mark.filterwarnings("ignore:Proactor event loop:RuntimeWarning")  # zmq on Windows
def test_jupyter(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("import json\n" + KERNEL),  # kernels defined in a cell
        nbformat.v4.new_code_cell(SYNC),
        nbformat.v4.new_code_cell(AWAIT_TOP),  # await at cell level (Jupyter's loop)
        nbformat.v4.new_code_cell("d"),  # affichage d'un gs.Array
    ]
    client = nbclient.NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}})
    client.execute()
    texts = []
    for cell in nb.cells:
        for o in cell.get("outputs", []):
            if o.get("name") == "stdout":
                texts.append(o["text"])
            elif "data" in o:
                texts.append(o["data"].get("text/plain", ""))
    out = "\n".join(texts)
    results = [json.loads(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("RESULT")]
    assert len(results) == 2, out
    for got in results:
        same(got, reference())
    assert "gs.Array(" in out
