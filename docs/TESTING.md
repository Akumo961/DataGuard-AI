# Testing Strategy

## Test layers

- Unit tests: domain models, PII detection, risk scoring, compliance evaluation, PIA transitions and audit integrity.
- API tests: authentication, validation, tenant boundary behavior, response redaction and health endpoints.
- Integration tests: document-processing pipeline and framework loading.
- Security regression tests: invalid authentication, malformed input, path traversal/archive safety and audit tampering.

## CI gates

GitHub Actions runs Ruff formatting/linting, mypy, pytest with coverage, and pip-audit on pushes to `main`/`production/**` and pull requests.

Coverage has a 60% minimum CI gate. This is an initial engineering floor, not a claim of comprehensive production assurance; critical security/privacy modules should have materially higher coverage as the product evolves.

## Test data

Tests must use synthetic or non-sensitive fixtures. Production personal information must never be committed to the repository or used as a test fixture.

## Limitations

A green CI run does not establish regulatory compliance, security certification, model quality, or readiness for government production deployment. Those require additional independent validation, threat testing, operational controls and legal/privacy review.
