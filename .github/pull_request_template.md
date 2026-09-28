# Pull Request Template

## Summary
<!-- Briefly describe the purpose of this PR. -->

## Related Issue
<!-- Link to the issue this PR addresses. Use keywords like `Closes #123` or `Fixes #123` to automate closing. -->

## Type of Change
Please check the type of change:
- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] 🚀 New feature (non-breaking change which adds functionality)
- [ ] 📝 Documentation update (changes to docs or examples)
- [ ] 🧹 Maintenance (refactoring, code cleanup, dependencies)
- [ ] ⚠️ Breaking change (fix or feature that would cause existing functionality to not work as expected)

## Changes Made
- [ ] Describe change 1
- [ ] Describe change 2

## Testing
<!-- How did you test these changes? Include commands and logs if applicable. -->
- [ ] All unit tests pass locally.
- [ ] Added new tests for the changes made.

## Trust Score Impact
<!-- Required when the change touches metrics, trust_score.py, the pipeline or the results contract. -->
- [ ] No effect on scores, **or** the reference models in `tests/reference/` and the characterization baselines were re-run and every change is listed below.
- [ ] Formula or threshold changes are reflected in `docs/trust_score_explained.md` and ADR-001 (`tests/invariants/test_formula_contract.py` passes).

| Reference model | Before | After | Why |
|---|---|---|---|
|  |  |  |  |

## Pre-submit Checklist
Before submitting, please ensure you have:
- [ ] Run `pre-commit run --all-files` and all hooks passed.
- [ ] Verified that all tests pass (`make test`).
- [ ] Updated the `CHANGELOG.md` if applicable.
- [ ] Verified that documentation has been updated for new features (`python -m sphinx -W -b html docs docs/_build/html` passes).

## Notes for Reviewers
<!-- Add any extra context or questions for the reviewers here. -->
