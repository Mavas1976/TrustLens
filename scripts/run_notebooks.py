"""Execute the example notebooks against the installed (checked-out) trustlens.

Cells that install packages (``!pip install`` / ``%pip install``) are skipped
so the notebooks run against the code under test, not a PyPI release, and
interactive matplotlib backends are replaced by ``%matplotlib inline``.

Usage: python scripts/run_notebooks.py [--write] [NOTEBOOK ...]
       (default: examples/*.ipynb; --write stores the fresh outputs in the notebooks)
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


def run(path: Path, timeout: int = 900, write: bool = False) -> None:
    nb = nbformat.read(path, as_version=4)
    original = [cell.source for cell in nb.cells]
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if any(m in cell.source for m in INSTALL_MARKERS):
            cell.source = "pass  # install cell skipped by scripts/run_notebooks.py"
        cell.source = re.sub(
            r"%matplotlib (widget|ipympl|notebook)", "%matplotlib inline", cell.source
        )
    with tempfile.TemporaryDirectory() as workdir:
        NotebookClient(
            nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": workdir}}
        ).execute()
    if write:
        # Keep the notebook as authored; only the outputs are refreshed.
        for cell, source in zip(nb.cells, original):
            if cell.cell_type == "code" and cell.source != source:
                cell.source = source
                if any(m in source for m in INSTALL_MARKERS):
                    cell.outputs = []
                    cell.execution_count = None
        nbformat.write(nb, path)


def main(argv: list[str]) -> int:
    write = "--write" in argv
    args = [a for a in argv if a != "--write"]
    paths = [Path(a) for a in args] or sorted((ROOT / "examples").glob("*.ipynb"))
    failed = []
    for path in paths:
        print(f"executing {path.name} ...", flush=True)
        try:
            run(path, write=write)
        except Exception as exc:  # report every notebook, then fail
            reason = f"{getattr(exc, 'ename', type(exc).__name__)}: {getattr(exc, 'evalue', exc)}"
            print(f"FAILED {path.name}: {reason}", flush=True)
            failed.append(path.name)
    print(f"{len(paths) - len(failed)}/{len(paths)} notebooks executed without errors")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
