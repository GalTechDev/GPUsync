#!/usr/bin/env python3
"""Documentation website (GitHub Pages): docs/markdown/*.md → docs/*.html, in the visual style of bio.galtech.cc.

    python3 tools/build_pages.py            # regenerates the site in docs/
    python3 tools/build_pages.py --check    # fails if the site is not up to date (tests, CI)

Standard library only. The Markdown converter covers what docs/ uses: headings, paragraphs, lists, tables, code
blocks, inline code, bold, italics, links. ```python blocks are marked "verified": tests/test_docs.py runs them.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "markdown"  # sources
OUT = ROOT / "docs"  # the site: GitHub Pages serves a branch from / or /docs only
REPO = "https://github.com/GalTechDev/GPUsync"
BRANCH = "new-rust"

# Menu order, icon (Material Symbols) and short title of each page
PAGES = [
    ("installation", "download", "Installation"),
    ("kernels", "code", "Writing a kernel"),
    ("results", "database", "Results and data"),
    ("devices", "developer_board", "Devices"),
    ("atomics", "scatter_plot", "Atomics"),
    ("groups", "hub", "Thread groups"),
    ("window", "desktop_windows", "Window"),
    ("errors", "bug_report", "Errors and performance"),
    ("reference", "menu_book", "Reference"),
]

FONTS = (
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;600;700&display=swap"
)
ICONS = "https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0..1,0&display=block"


# ───────────────────────── Code highlighting ─────────────────────────

_PY = re.compile(
    r"(?P<com>#[^\n]*)"
    r"|(?P<str>\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''|[rbf]?\"(?:\\.|[^\"\\\n])*\"|[rbf]?'(?:\\.|[^'\\\n])*')"
    r"|(?P<dec>^[ \t]*@[\w.]+)"
    r"|(?P<gs>\bgs\.[A-Za-z_]\w*)"
    r"|(?P<kw>\b(?:def|return|for|in|if|elif|else|while|import|from|as|with|try|except|finally|raise|lambda|and|or"
    r"|not|is|None|True|False|async|await|break|continue|pass|class|yield|global|nonlocal|del|assert)\b)"
    r"|(?P<num>\b(?:0[xob][\da-fA-F_]+|\d[\d_]*(?:\.\d*)?(?:[eE][+-]?\d+)?)\b)",
    re.M,
)
_SH = re.compile(r"(?P<com>(?:^|(?<=\s))#[^\n]*)|(?P<str>\"(?:\\.|[^\"\\\n])*\"|'[^'\n]*')", re.M)


def highlight(code: str, lang: str) -> str:
    pattern = _PY if lang in ("python", "py") else _SH if lang in ("bash", "sh") else None
    if pattern is None:
        return html.escape(code, quote=False)
    out, pos = [], 0
    for m in pattern.finditer(code):
        out.append(html.escape(code[pos : m.start()], quote=False))
        out.append(f'<span class="tk-{m.lastgroup}">{html.escape(m.group(), quote=False)}</span>')
        pos = m.end()
    out.append(html.escape(code[pos:], quote=False))
    return "".join(out)


def code_block(code: str, lang: str, *, show_badge: bool = True) -> str:
    label = {"python": "Python", "py": "Python", "bash": "Terminal", "text": "Output"}.get(lang, lang or "Code")
    if lang == "python":
        badge = '<span class="badge badge-ok" title="Run by tests/test_docs.py">Verified</span>'
    elif lang == "py":
        badge = '<span class="badge badge-muted" title="Excerpt not run by the tests">Excerpt</span>'
    else:
        badge = ""
    if not show_badge:
        badge = ""
    copy = '<button class="copy" type="button">Copy</button>' if lang != "text" else ""
    return (
        f'<div class="code"><div class="code-head"><span>{label}</span>{badge}<span class="spacer"></span>{copy}</div>'
        f"<pre><code>{highlight(code.rstrip(), lang)}</code></pre></div>"
    )


# ───────────────────────── Markdown ─────────────────────────


def slug(text: str) -> str:
    import unicodedata

    t = unicodedata.normalize("NFKD", re.sub(r"<[^>]+>", "", text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-") or "section"


def link_target(url: str) -> str:
    """Links between docs/ pages → site pages; to the rest of the repository → GitHub."""
    if re.match(r"^[a-z]+://", url) or url.startswith("#"):
        return url
    path, _, frag = url.partition("#")
    frag = f"#{frag}" if frag else ""
    if "/" not in path and path.endswith(".md"):
        name = path[:-3]
        return ("index.html" if name == "README" else f"{name}.html") + frag
    if path.startswith("../../"):
        rel = path[6:]
        kind = "tree" if rel.endswith("README.md") and "/" in rel else "blob"
        if kind == "tree":
            rel = rel.rsplit("/", 1)[0]
        return f"{REPO}/{kind}/{BRANCH}/{rel}{frag}"
    return url


def inline(text: str) -> str:
    """Inline code, links, bold, italics; everything else is escaped."""
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(f"<code>{html.escape(p[1:-1], quote=False)}</code>")
            continue
        s = html.escape(p, quote=False)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
        out.append(s)
    s = "".join(out)
    # Links last: their text may contain code
    return re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)\)",
        lambda m: f'<a href="{html.escape(link_target(html.unescape(m.group(2))))}">{m.group(1)}</a>',
        s,
    )


def split_row(line: str) -> list[str]:
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|") if not line.strip().endswith("\\|") else line.strip()[1:])
    return [c.strip().replace("\\|", "|") for c in cells]


def markdown(text: str) -> tuple[str, str, str, list[tuple[str, str]]]:
    """Renders a document: (title, lead, HTML body, table of contents [(anchor, title)])."""
    lines = text.split("\n")
    title, lead, body, toc = "", "", [], []
    i, n = 0, len(lines)
    para: list[str] = []

    def flush():
        nonlocal lead
        if para:
            content = inline(" ".join(x.strip() for x in para))
            if not lead and not body:
                lead = content
            else:
                body.append(f"<p>{content}</p>")
            para.clear()

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            lang = stripped[3:].strip()
            j = i + 1
            while j < n and not lines[j].strip().startswith("```"):
                j += 1
            body.append(code_block("\n".join(lines[i + 1 : j]), lang))
            i = j + 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            flush()
            level, content = len(m.group(1)), m.group(2).strip()
            if level == 1 and not title:
                title = content
            else:
                anchor = slug(content)
                if level == 2:
                    toc.append((anchor, inline(content)))
                body.append(
                    f'<h{level} id="{anchor}"><a class="anchor" href="#{anchor}" aria-hidden="true">#</a>{inline(content)}</h{level}>'
                )
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            flush()
            head = split_row(line)
            rows = []
            j = i + 2
            while j < n and lines[j].strip().startswith("|"):
                rows.append(split_row(lines[j]))
                j += 1
            th = "".join(f"<th>{inline(c)}</th>" for c in head) if any(head) else ""
            tr = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows)
            thead = f"<thead><tr>{th}</tr></thead>" if th else ""
            body.append(f'<div class="table-wrap"><table>{thead}<tbody>{tr}</tbody></table></div>')
            i = j
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m and not m.group(1):
            flush()
            ordered = m.group(2)[0].isdigit()
            items: list[list[str]] = []
            j = i
            while j < n:
                mm = re.match(r"^([-*]|\d+\.)\s+(.*)$", lines[j])
                if mm and (mm.group(1)[0].isdigit()) == ordered:
                    items.append([mm.group(2)])
                elif lines[j].startswith("  ") and lines[j].strip() and items:
                    items[-1].append(lines[j].strip())
                else:
                    break
                j += 1
            tag = "ol" if ordered else "ul"
            lis = "".join(f"<li>{inline(' '.join(it))}</li>" for it in items)
            body.append(f"<{tag}>{lis}</{tag}>")
            i = j
            continue
        if not stripped:
            flush()
        else:
            para.append(line)
        i += 1
    flush()
    return title, lead, "\n".join(body), toc


# ───────────────────────── Template ─────────────────────────


def nav(current: str) -> str:
    items = [f'<a href="index.html"{_cur(current, "index")}><span class="icon">home</span>Home</a>']
    items.append('<div class="nav-group">Guide</div>')
    for name, icon, short in PAGES:
        if name == "reference":
            items.append('<div class="nav-group">API</div>')
        items.append(f'<a href="{name}.html"{_cur(current, name)}><span class="icon">{icon}</span>{html.escape(short)}</a>')
    return "\n".join(items)


def _cur(current: str, name: str) -> str:
    return ' aria-current="page"' if current == name else ""


def layout(*, name: str, title: str, top: str, description: str, main: str, toc: str = "", version: str) -> str:
    wide = "" if toc else " wide"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<link rel="stylesheet" href="{ICONS}">
<link rel="stylesheet" href="assets/style.css">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
<script src="assets/site.js"></script>
</head>
<body>
<aside class="side" aria-label="Navigation">
<a class="brand" href="index.html">GPUsync <small>v{version}</small></a>
<nav class="nav">
{nav(name)}
</nav>
<div class="side-foot">MIT licence · <a href="{REPO}/tree/{BRANCH}">Source code</a></div>
</aside>
<div class="scrim"></div>
<div class="page">
<header class="top">
<button class="menu-toggle" type="button" aria-label="Menu" aria-expanded="false"><span class="icon">menu</span></button>
<h1 class="top-title">{html.escape(top)}</h1>
<button class="btn btn-plain theme-toggle" type="button" aria-label="Theme"><span class="icon">dark_mode</span></button>
<a class="btn" href="{REPO}/tree/{BRANCH}"><span class="label">GitHub</span><span class="icon">terminal</span></a>
</header>
<main class="content{wide}">
{main}
{toc}
</main>
<footer class="footer"><span>GPUsync · GPU kernels in Python · CUDA, Metal, Vulkan, DX12, OpenCL</span><span>Generated from docs/markdown by tools/build_pages.py</span></footer>
</div>
</body>
</html>
"""


def doc_page(name: str, md: str, version: str) -> str:
    title, lead, body, toc = markdown(md)
    short = next(s for n, _, s in PAGES if n == name)
    order = [p[0] for p in PAGES]
    k = order.index(name)
    prev = PAGES[k - 1] if k > 0 else None
    nxt = PAGES[k + 1] if k + 1 < len(PAGES) else None
    pager = '<nav class="pager">'
    if prev:
        pager += f'<a href="{prev[0]}.html"><span class="dir">← Previous</span><span class="name">{html.escape(prev[2])}</span></a>'
    if nxt:
        pager += f'<a class="next" href="{nxt[0]}.html"><span class="dir">Next →</span><span class="name">{html.escape(nxt[2])}</span></a>'
    pager += "</nav>"
    head = f'<h2 class="lead-title">{inline(title)}</h2>' + (f'<p class="lead">{lead}</p>' if lead else "")
    main = f'<article class="doc">{head}<hr class="rule">{body}{pager}</article>'
    toc_html = ""
    if len(toc) >= 2:
        links = "".join(f'<a href="#{a}">{t}</a>' for a, t in toc)
        toc_html = f'<aside class="toc" aria-label="On this page"><div class="toc-title">On this page</div>{links}</aside>'
    plain = re.sub(r"<[^>]+>", "", lead)
    return layout(
        name=name, title=f"{re.sub(r'<[^>]+>', '', inline(title))} — GPUsync", top=short,
        description=html.unescape(plain)[:200], main=main, toc=toc_html, version=version,
    )


HOME_CODE = '''import numpy as np
import gpusync as gs

@gs.kernel
def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
    return (x - mean) / std

x = np.random.rand(1_000_000).astype(np.float32)
y = normalize(x, x.mean(), x.std())     # runs on the GPU
print(y[:5])                            # waits for the result and copies it back'''

FEATURES = [
    ("Kernels in Python", "Native",
     "You write how one cell of the result is computed, with simple types. GPUsync translates it and runs it for every cell in parallel.",
     ["@gs.kernel", "gs.index()", "gs.f32", "gs.i64"]),
    ("Every GPU", "Portable",
     "The same code on an NVIDIA card, a Mac or a laptop with an integrated GPU. The device is chosen from what the kernel needs.",
     ["CUDA", "METAL", "VULKAN", "DX12", "OPENCL"]),
    ("No size limit", "Out-of-core",
     "Inputs and outputs larger than GPU memory, or than main memory: the computation is split and transfers are overlapped.",
     ["np.memmap", "out=", ".stream()"]),
    ("Async and interop", "Modern",
     "Results stay on the GPU until they are read. await, Jupyter, several threads, zero-copy exchange through DLPack.",
     ["ASYNCIO", "DLPACK", "PYTORCH"]),
    ("Groups and atomics", "Advanced",
     "Shared memory, barriers checked at compile time, atomic writes, Tensor Cores for byte matrix products.",
     ["gs.shared", "gs.barrier", "gs.atomic_add"]),
    ("Native window", "Real time",
     "An image computed on the GPU goes straight to the screen, without going through main memory.",
     ["gs.Window", "ZERO COPY"]),
]

STATS = [
    ("5", "backends: CUDA, Metal, Vulkan, DX12, OpenCL"),
    ("× 45", "faster than NumPy: histogram of 100 million values (Apple M5 Pro)"),
    ("100 % portable", "the same code on every machine, no GPU SDK to install: the drivers already there are enough"),
    ("0 copies", "on unified-memory GPUs"),
]


def _sentence(text: str) -> str:
    text = text.strip()
    return (text[:1].upper() + text[1:] + ("" if text.endswith(".") else ".")) if text else text


def home(descriptions: dict[str, str], version: str) -> str:
    def card(title, badge, text, tags, href=None):
        tags_html = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in tags)
        head = f'<div class="card-head"><h3 class="card-title">{html.escape(title)}</h3><span class="badge badge-ok">{html.escape(badge)}</span></div>'
        body = f'<div class="card-body"><p>{html.escape(text)}</p>' + (f'<div class="tags">{tags_html}</div>' if tags else "") + "</div>"
        return f'<a class="card" href="{href}">{head}{body}</a>' if href else f'<div class="card">{head}{body}</div>'

    features = "".join(card(*f) for f in FEATURES)
    docs_cards = "".join(
        card(short, f"{k + 1:02d}", _sentence(re.sub(r"<[^>]+>", "", html.unescape(descriptions.get(name, "")))), [], f"{name}.html")
        for k, (name, _, short) in enumerate(PAGES)
    )
    stats = "".join(f'<div class="stat"><div class="stat-value">{html.escape(v)}</div><div class="stat-label">{html.escape(l)}</div></div>' for v, l in STATS)
    main = f"""<div class="home">
<section class="hero">
<div>
<h2 class="hero-title">GPUsync</h2>
<p class="hero-sub">Write GPU kernels in Python, and run them on any GPU.</p>
<div class="hero-actions">
<a class="btn btn-primary" href="installation.html"><span class="icon">download</span>Install</a>
<a class="btn" href="kernels.html"><span class="icon">code</span>First kernel</a>
</div>
</div>
<div class="callout">{code_block(HOME_CODE, "py", show_badge=False)}</div>
</section>
<div class="stats">{stats}</div>
<h2 class="section-title">What GPUsync does</h2>
<p class="section-lead">A Python library with a Rust core, and no GPU dependency to install.</p>
<div class="cards">{features}</div>
<h2 class="section-title">Documentation</h2>
<p class="section-lead">Every Python example is run by the tests: marked <span class="badge badge-ok">Verified</span>, it works as written.</p>
<div class="cards">{docs_cards}</div>
<h2 class="section-title">Graphics engine</h2>
<p class="section-lead">2D and 3D rendering, fluids, atmosphere and weather: the separate <code>gpusync-gfx</code> package, written with GPUsync.</p>
<div class="cards">{card("gpusync-gfx", "Optional", "Installed on top of the library, it is imported as gpusync.gfx. Without it, GPUsync works normally.", ["pip install gpusync-gfx", "import gpusync.gfx"], f"{REPO}/tree/{BRANCH}/gfx")}</div>
</div>"""
    return layout(
        name="index", title="GPUsync — GPU kernels in Python", top="Overview",
        description="Write GPU kernels in Python and run them on CUDA, Metal, Vulkan, DX12 and OpenCL.",
        main=main, version=version,
    )


FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" fill="#006df8"/><path d="M9 11h14M9 16h9M9 21h14" stroke="#fefcff" stroke-width="2.5" stroke-linecap="square"/></svg>
"""


def build() -> dict[Path, str]:
    """Contents of every generated file."""
    version = re.search(r'^version = "([^"]+)"', (ROOT / "Cargo.toml").read_text(encoding="utf-8"), re.M).group(1)
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    descriptions = {}
    for m in re.finditer(r"^\|\s*\[[^\]]+\]\(([a-z]+)\.md\)\s*\|\s*(.*?)\s*\|\s*$", index, re.M):
        descriptions[m.group(1)] = inline(m.group(2))
    files = {OUT / "index.html": home(descriptions, version), OUT / "assets" / "favicon.svg": FAVICON, OUT / ".nojekyll": ""}
    for name, _, _ in PAGES:
        files[OUT / f"{name}.html"] = doc_page(name, (DOCS / f"{name}.md").read_text(encoding="utf-8"), version)
    return files


def _orphans(files: dict[Path, str]) -> list[Path]:
    """Generated pages left over from docs that were renamed or removed."""
    return sorted(p for p in OUT.glob("*.html") if p not in files)


def main() -> int:
    files = build()
    if "--check" in sys.argv:
        stale = [p for p, text in files.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        stale += _orphans(files)
        if stale:
            print("the site in docs/ is not up to date, run: python3 tools/build_pages.py")
            for p in stale:
                print("  ", p.relative_to(ROOT))
            return 1
        print("the site in docs/ is up to date")
        return 0
    for p in _orphans(files):
        p.unlink()
    for p, text in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    print(f"{len(files)} files written to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
