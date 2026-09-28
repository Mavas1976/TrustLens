# Working agreement for agents on this repository

This fork (`Mavas1976/TrustLens`) uses a **solo-developer / agent workflow**.
The repository owner set this rule; it applies until they say otherwise.

## Change workflow

1. Work on a feature branch, never directly on `main`.
2. Open a pull request against `main`.
3. Run the local checks below before every push; all must exit 0 (judge by
   exit code, not by filtering output).
4. Wait for **every required CI check** on the PR's head commit to finish.
5. If all required checks pass and the PR is mergeable, **merge it yourself**
   (squash merge, matching the existing history) and **delete the feature
   branch**.
6. Human review or approval is **not** a merge requirement. Request one only
   when the owner explicitly asks for it.

## Never

- Merge with a failing, pending, cancelled or missing required check. If CI
  did not run at all (for example because GitHub Actions is disabled), that
  is a blocker to report, not a pass.
- Skip, disable or quarantine a test, loosen a check, or push an empty commit
  to get green.
- Enable GitHub auto-merge while no required checks are configured: it would
  merge before CI has run.

## Required checks

The jobs in `.github/workflows/ci.yml` (lint, typecheck, tests matrix,
dependency-compat, security, docs, examples, build) and CodeQL
(`.github/workflows/codeql.yml`). The tests job also runs the docstring
examples (`pytest trustlens --doctest-modules --import-mode=importlib`).
When a PR touches `examples/*.ipynb`, the Notebooks workflow
(`.github/workflows/notebooks.yml`) is required too; locally:
`pip install -e ".[notebooks]" && python scripts/run_notebooks.py`.

## Local checks (run before pushing)

```bash
ruff check . && ruff format --check .
mypy trustlens/
pre-commit run --all-files
MPLBACKEND=Agg python -m pytest -p no:cacheprovider -q
MPLBACKEND=Agg python -m pytest trustlens --doctest-modules --import-mode=importlib -q
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

## Project conventions

- Trust Score methodology changes need an ADR update
  (`docs/adr/ADR-001-trust-score-methodology.md`), a CHANGELOG entry and the
  "Trust Score Impact" section of the PR template. The published table in
  `docs/trust_score_explained.md` is enforced by
  `tests/reference/test_published_table.py`.
- Regenerate characterization baselines with
  `python tests/characterization/generate_baselines.py`, never by hand.
