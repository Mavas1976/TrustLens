"""Execute the example notebooks against the installed (checked-out) trustlens.

Cells that install packages (``!pip install`` / ``%pip install``) are skipped
so the notebooks run against the code under test, not a PyPI release, and
interactive matplotlib backends are replaced by ``%matplotlib inline``.

Usage: python scripts/run_notebooks.py [NOTEBOOK ...]   (default: examples/*.ipynb)
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
INSTALL_MARKERS = ("!pip install", "%pip install")


def run(path: Path, timeout: int = 900) -> None:
    nb = nbformat.read(path, as_version=4)
    nb.cells = [
        cell
        for cell in nb.cells
        if not (cell.cell_type == "code" and any(m in cell.source for m in INSTALL_MARKERS))
    ]
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = re.sub(
                r"%matplotlib (widget|ipympl|notebook)", "%matplotlib inline", cell.source
            )
    with tempfile.TemporaryDirectory() as workdir:
        NotebookClient(
            nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": workdir}}
        ).execute()


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or sorted((ROOT / "examples").glob("*.ipynb"))
    failed = []
    for path in paths:
        print(f"executing {path.name} ...", flush=True)
        try:
            run(path)
        except Exception as exc:  # report every notebook, then fail
            reason = f"{getattr(exc, 'ename', type(exc).__name__)}: {getattr(exc, 'evalue', exc)}"
            print(f"FAILED {path.name}: {reason}", flush=True)
            failed.append(path.name)
    print(f"{len(paths) - len(failed)}/{len(paths)} notebooks executed without errors")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
