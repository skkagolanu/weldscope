# CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment.

## Workflows

### CI Pipeline (`.github/workflows/ci.yml`)

Runs on every push and pull request:

**Test Job:**
- Runs on Python 3.11 and 3.12
- Executes full test suite with pytest
- Enforces 90% minimum coverage
- Uploads coverage to Codecov

**Lint Job:**
- Code formatting check with Black
- Static analysis with Pylint
- Type checking with MyPy

**Security Job:**
- Security scan with Bandit
- Dependency vulnerability check with Safety

### PR Quality Gate (`.github/workflows/pr-gate.yml`)

Blocks PR merges if:
- Tests fail
- Coverage < 90%
- Code not formatted
- Security issues found

### Deployment (`.github/workflows/deploy.yml`)

Auto-deploys to Streamlit Cloud on main branch push.

## Local Development

### Initial Setup

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install git hooks
./setup-hooks.sh
```

### Before Pushing

**Always run local quality checks:**

```bash
./check-quality.sh
```

This will:
1. Format code with Black
2. Run full test suite (90% coverage required)
3. Lint with Pylint
4. Security scan with Bandit

**Or run individually:**

```bash
# Format
black weldscope/ tests/

# Test
pytest tests/ -v --cov=weldscope --cov-fail-under=90

# Lint
pylint weldscope/ --disable=C0114,C0115,C0116,R0913,R0914

# Security
bandit -r weldscope/ -ll
```

### Git Hooks

Pre-commit hook automatically runs:
- Code formatting check
- Full test suite
- Security scan

**Skip hooks (not recommended):**
```bash
git commit --no-verify
```

### Run Tests
```bash
pytest tests/ -v --cov=weldscope
```

### Format Code
```bash
black weldscope/ tests/
```

### Lint
```bash
pylint weldscope/
```

### Security Scan
```bash
bandit -r weldscope/ -ll
```

### Full Quality Check
```bash
# Run all checks locally before pushing
pytest tests/ --cov=weldscope --cov-fail-under=90
black --check weldscope/ tests/
pylint weldscope/ --disable=C0114,C0115,C0116,R0913,R0914
bandit -r weldscope/ -ll
```

## Required Secrets

Configure in GitHub repository settings:

- `HF_TOKEN`: HuggingFace API token (required for tests)
- `CODECOV_TOKEN`: Codecov upload token (optional)

## Branch Protection Rules

Recommended settings for `main` branch:

- Require pull request reviews (1 approver)
- Require status checks to pass
  - `test (3.11)`
  - `test (3.12)`
  - `lint`
  - `security`
- Require branches to be up to date
- Require conversation resolution
- Do not allow force pushes
- Do not allow deletions

## Coverage Reports

Coverage reports are:
- Displayed in PR comments
- Uploaded to Codecov
- Generated as HTML in `htmlcov/`
- Included in CI logs

## Troubleshooting

### Tests fail locally but pass in CI
- Check Python version (use 3.11 or 3.12)
- Verify all dependencies installed: `pip install -r requirements-dev.txt`
- Clear pytest cache: `pytest --cache-clear`

### Coverage below threshold
- Run with missing lines: `pytest --cov=weldscope --cov-report=term-missing`
- Add tests for uncovered code
- Update threshold in `pyproject.toml` if justified

### Black formatting issues
- Auto-fix: `black weldscope/ tests/`
- Check config in `pyproject.toml`

### Security scan failures
- Review Bandit output for severity
- Add `# nosec` comment with justification if false positive
- Update vulnerable dependencies
