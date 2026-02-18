## Summary

<!-- What does this PR do? Keep it to 2–4 bullet points. -->

-
-

## Type of Change

<!-- Check all that apply -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that changes existing behaviour)
- [ ] Documentation update
- [ ] Refactor (no functional change)
- [ ] Dependency update

## Related Issue

<!-- Link the issue this PR resolves, if any -->

Closes #

## Changes Made

<!-- Bullet-point list of files changed and what was changed. Be specific. -->

- `agents/my_agent.py` — Added `MyAgent` class with …
- `storage/database.py` — Added `my_table` schema and `log_my_result()` method
- `tests/test_my_agent.py` — Unit tests for …

## Testing

<!-- Describe how you tested the changes -->

- [ ] Ran existing test suite: `python3 -m pytest tests/ -v` — all pass
- [ ] Added new tests for new logic
- [ ] Tested manually: `python main.py AAPL` completes without errors
- [ ] Tested the dashboard: `streamlit run dashboard/app.py` — no regressions

**Manual test commands run:**

```bash
# paste the commands you ran to verify your changes
python main.py AAPL
python3 -m pytest tests/ -v
```

## Checklist

- [ ] My code follows the conventions in [docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md)
- [ ] I have added type hints to all new functions
- [ ] I have added docstrings to all new public methods
- [ ] `CHANGELOG.md` has been updated under `[Unreleased]`
- [ ] No secrets or `.env` files are committed (`git status` is clean)
- [ ] `requirements.txt` is updated if new packages were added
- [ ] The PR branch is up to date with `main`

## Screenshots / Output

<!-- If the PR changes the dashboard or terminal output, paste a screenshot or output snippet -->

```
# example output
```
