# Contributing

Thanks for your interest in improving **Customer Churn Prediction**! Contributions of all kinds —
bug reports, features, docs, and tests — are welcome.

## Getting Started

1. **Fork** the repository and clone your fork.
2. Create a virtual environment and install the development dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
   python -m pip install -r requirements-dev.txt
   ```
3. Create a branch for your change:
   ```bash
   git checkout -b feat/short-description
   ```

## Development Workflow

Before opening a pull request, make sure the same checks that run in CI pass locally:

```bash
ruff check .          # lint
ruff format --check . # formatting
pytest                # test suite
```

## Pull Requests

- Keep changes focused and scoped to a single concern.
- Add or update tests for any behavior change.
- Update the README or docstrings when behavior or interfaces change.
- Write a clear PR description and link any related issues.

## Commit Messages

Use clear, present-tense messages (e.g. `Add batch prediction endpoint`). Conventional Commit
prefixes such as `feat:`, `fix:`, and `docs:` are appreciated but not required.

## Code of Conduct

By participating, you agree to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).
