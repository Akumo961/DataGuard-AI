# Phase 25 — Enterprise Release Baseline

## Objective

Establish a single, auditable baseline for the DataGuard Québec release candidate after the extended roadmap. This phase prevents branch/documentation drift and defines what must be true before a release is presented to an enterprise or government evaluator.

## Baseline principles

1. The release candidate is identified by an immutable commit SHA and version.
2. Every release artifact must be traceable to source, dependencies and CI evidence.
3. A documented capability is not treated as production evidence unless it has been technically verified.
4. Legal/compliance mappings require qualified review.
5. AI/ML claims require reproducible evaluation evidence.
6. Synthetic demonstration data is never evidence of production data quality or security.

## Release record

For each candidate, record:

- version
- commit SHA
- build timestamp
- Python/runtime versions
- dependency lock/hash information
- database migration revision
- model/rule versions
- framework/control-content versions
- CI run identifiers
- security scan results
- SBOM location
- approval owner
- release decision

## Required release evidence

### Code and tests

- Formatting/lint/type checks pass.
- Unit/integration/API/security suites pass.
- Regression suite passes.
- Known failures are documented and risk-accepted where applicable.

### Security

- Dependency/container vulnerabilities reviewed.
- Secrets are not embedded in source or images.
- Production CORS/headers/TLS configuration is validated.
- Authentication and authorization are tested.
- Tenant isolation is tested adversarially.
- File-processing boundaries are validated.

### Data and privacy

- Retention/deletion configuration is tested.
- Audit evidence is traceable and protected.
- Production data residency and cross-border processing are documented.
- Sensitive-data handling is minimized and access-controlled.

### AI/ML

- Model/rule provenance is recorded.
- Evaluation dataset/version is recorded.
- Precision, recall, F1 and error analysis are reproducible.
- Human-review behavior is tested.

### Operations

- Health/readiness checks work.
- Logging/metrics/alerts are configured.
- Backup and restoration have evidence.
- Disaster recovery has evidence.
- Incident and rollback procedures are documented.

## Go/no-go policy

A release is **GO** only when mandatory evidence is approved by the designated owner. A release is **NO-GO** for unresolved critical security defects, failed tenant isolation, unavailable required identity controls, failed backup restoration, unverified mandatory privacy requirements, or unsupported legal/compliance claims.

## Enterprise evaluator package

A buyer-facing package should contain the product overview, architecture, security architecture, threat model, privacy architecture, ML validation, API/deployment documentation, PIA guide, audit/evidence description, demo script, procurement package and this release baseline.

## Phase 25 completion

The release-governance baseline is now defined in the repository. Execution of target-environment tests, independent assessments and organizational approvals remains external evidence and must not be fabricated by the repository.
