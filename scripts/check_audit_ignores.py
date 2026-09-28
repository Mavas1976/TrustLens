"""Validate the pip-audit ignore list and emit the matching command-line flags.

Usage:
    python scripts/check_audit_ignores.py FILE           # validate only
    python scripts/check_audit_ignores.py --flags FILE   # validate, print --ignore-vuln flags

Each non-comment line must read ``<id> | expires <YYYY-MM-DD> | <reason>``.
The script exits with status 1 when a line is malformed, has an empty
reason, or has expired, so CI forces ignored vulnerabilities to be
re-reviewed (audit issue TL-24).
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

LINE = re.compile(
    r"^(?P<id>[A-Z]+-[0-9]{4}-[0-9]+)\s*\|\s*expires (?P<date>\d{4}-\d{2}-\d{2})\s*\|\s*(?P<reason>.*)$"
)


def parse(path: Path, today: dt.date) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    errors: list[str] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = LINE.match(line)
        if not match:
            errors.append(f"line {number}: expected '<id> | expires YYYY-MM-DD | <reason>'")
            continue
        if not match["reason"].strip():
            errors.append(f"line {number}: {match['id']} has no reason")
        if dt.date.fromisoformat(match["date"]) < today:
            errors.append(f"line {number}: {match['id']} expired on {match['date']}; re-review it")
        ids.append(match["id"])
    return ids, errors


def main(argv: list[str]) -> int:
    emit_flags = "--flags" in argv
    paths = [a for a in argv if a != "--flags"]
    if len(paths) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    ids, errors = parse(Path(paths[0]), dt.date.today())
    for error in errors:
        print(f"pip-audit ignore list: {error}", file=sys.stderr)
    if errors:
        return 1
    if emit_flags:
        print(" ".join(f"--ignore-vuln {vid}" for vid in ids))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
