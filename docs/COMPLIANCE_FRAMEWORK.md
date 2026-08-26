# Compliance Framework Engine

DataGuard provides a versioned, machine-readable control framework mechanism. It is a governance/evidence tool and does **not** determine whether an organization is legally compliant.

## Frameworks

- `compliance/frameworks/quebec_privacy.yaml`
- `compliance/frameworks/canada_privacy.yaml`
- `compliance/frameworks/gdpr.yaml`
- `compliance/frameworks/ccpa.yaml`

Each rule contains an ID, title, description, category, severity, evidence requirements, assessment questions, remediation guidance, version, source/reference metadata and applicability.

## Safety model

Framework files are marked `verification_required`. References intentionally identify the need for legal verification rather than inventing statutory citations. Before production use, qualified counsel/privacy professionals should verify every control against the current applicable legislation, sector obligations, regulator guidance and organizational policy.

## Evaluation

The engine compares configured evidence against required evidence. Results are `PASS`, `REQUIRES_REMEDIATION`, or `REQUIRES_REVIEW`. A `REQUIRES_REVIEW` result is used when applicability itself needs human/legal assessment.

## Versioning

Framework versions are explicit in each YAML document. Framework updates should be reviewed, tested, approved and released as controlled changes; existing assessments should retain the framework version used at assessment time.
