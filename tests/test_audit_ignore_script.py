"""scripts/check_audit_ignores.py enforces reasons and expiry dates (TL-24)."""

import datetime as dt
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "check_audit_ignores", ROOT / "scripts/check_audit_ignores.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_valid_entries_pass(tmp_path):
    f = tmp_path / "ignore.txt"
    f.write_text("# comment\nPYSEC-2025-1 | expires 2099-01-01 | not reachable from our code\n")
    ids, errors = module.parse(f, dt.date(2026, 9, 28))
    assert ids == ["PYSEC-2025-1"] and errors == []


def test_expired_missing_reason_and_malformed_entries_fail(tmp_path):
    f = tmp_path / "ignore.txt"
    f.write_text(
        "CVE-2025-1 | expires 2020-01-01 | old\nCVE-2025-2 | expires 2099-01-01 | \nCVE-2025-3\n"
    )
    _, errors = module.parse(f, dt.date(2026, 9, 28))
    assert len(errors) == 3
    assert module.main([str(f)]) == 1


def test_repository_ignore_list_is_valid():
    _, errors = module.parse(ROOT / ".github/pip-audit-ignore.txt", dt.date.today())
    assert errors == []
