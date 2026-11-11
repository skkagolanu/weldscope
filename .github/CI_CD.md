# Continuous checks

All application jobs target Python 3.12. Runtime, development and documentation dependencies are pinned with compatible transitive versions in `constraints.txt`.

- `ci.yml`: tests with a 90% coverage floor, formatting, Bandit, and advisory lint/type checks. Unit/integration tests need no real inference credentials.
- `pr-gate.yml`: tests, coverage, formatting and Bandit for pull requests.
- `deploy.yml`: browser checks after successful CI, daily, and manually. It detects the embedded app frame and verifies a local greeting. A failure produces screenshot/text artifacts.

Streamlit deploys the repository itself; a GitHub check is not a deployment mechanism or evidence that external generation works. `STREAMLIT_APP_URL` can override the default `https://weldscope.streamlit.app/` for the hosted check.

Run `pytest tests/ --cov-fail-under=90`, `black --check weldscope tests app.py scripts examples`, and `pip check` locally. Optional model-judged evaluations are separate, explicit provider calls, not part of routine credential-free CI.
