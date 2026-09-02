# DataGuard — Bank-Grade Acquisition Technical Gap Analysis

**Baseline reviewed:** `main` at `9346da5ed987f79ea0b4131ed393247b8d75f18d` (2026-09-01)

This document is an evidence-based engineering gap analysis. It does not treat documentation as proof of implementation.

## Executive assessment

DataGuard already has a meaningful privacy-platform foundation: FastAPI APIs, tenant-scoped PostgreSQL models, RLS migrations, centralized RBAC policy, JWT/OIDC validation primitives, layered regex + optional NER detection, explainable risk scoring, configurable compliance rules, PIA state transitions, append-only/tamper-evident audit primitives, document validation/extraction, Docker, and CI/SBOM workflows.

It is **not yet a credible $9M–$15M acquisition target on technology alone**. The largest current blockers are implementation depth and evidence: the enterprise API surface is narrow, classification is only an abstract contract, connectors are not implemented, asynchronous workers are only an interface, governance/remediation APIs are incomplete, production identity lifecycle is delegated rather than integrated, operational observability is thin, and the latest quality workflow is failing at the current baseline.

## Implementation matrix

| Capability | Actually implemented | Partial | Documentation only | Broken / risk | Acquisition importance |
|---|---:|---:|---:|---|---|
| FastAPI versioned API | Yes | | | API surface is narrow | High |
| Authentication | JWT validation; local dev auth | OIDC validation is a primitive, no provider integration/lifecycle | | Production identity depends on external deployment | Critical |
| Centralized RBAC | Policy + route dependencies | Permission/resource coverage is incomplete | | Unknown roles can still create dependency-level failures if introduced outside decoder | Critical |
| Tenant isolation | Tenant predicates + PostgreSQL RLS migrations | Not every workflow is exposed/tested | | Auth registration/login must establish tenant context under FORCE RLS | Critical |
| Document security | Extension/magic/MIME checks, OOXML limits, request-size middleware | Malware scanning/isolation is deployment responsibility | | Some parser failures are not normalized into safe API errors | Critical |
| PII detection | Regex/context + optional spaCy NER | Benchmarking and human review are limited | | No broad enterprise source discovery | High |
| Classification | Contract only | No concrete classifier wired into analysis | | Product cannot yet prove enterprise classification capability | Critical |
| Risk engine | Deterministic explainable scoring | Limited context dimensions and policy model | | Not yet a full enterprise risk model | High |
| Regulatory mapping | YAML framework loader + evaluator | Evidence model/applicability workflow is basic | | Compliance output must not be mistaken for legal compliance | High |
| PIA | State machine + persistence | API workflow lacks rich inventory/vendor/jurisdiction/approval evidence model | | No complete enterprise PIA workbench | Critical |
| Remediation | Create endpoint + persistence | No complete lifecycle/listing/evidence/approval API | | Findings are not consistently converted into governed work | Critical |
| Audit | Hash chain + append-only DB trigger + verification endpoint | Event coverage and actor/request metadata are incomplete | | Some audit events use `actor_id=None`, `request_id=None`, `ip_address=None` | Critical |
| Enterprise API | `/api/v1` and OpenAPI via FastAPI | No mature resources for inventory, findings, policies, org administration, integrations | | Narrow API limits integration value | Critical |
| Connectors | Ingestion abstractions | No S3/Azure/SharePoint/Drive/database/SIEM/IAM/ticketing adapters | Documentation/roadmap references only | Missing | Critical |
| Async processing | Job/queue abstract interface | No concrete Redis worker/queue implementation | Documentation/roadmap | Missing | Critical |
| Frontend | Responsive static dashboard | Most enterprise views are placeholders and session-local | | Product workflow is materially incomplete | High |
| Observability | Structured logging dependency + basic request IDs | No complete metrics/traces/security-event operational dashboard | Documentation | Partial | High |
| CI/CD | Ruff, mypy, pytest, pip-audit, Docker E2E, SBOM workflows | Container scanning/secret scanning and broader gates need verification | | Latest quality workflow is failing | Critical |
| Deployment | Docker + PostgreSQL + Redis | No Kubernetes/Helm/private-cloud/on-prem delivery package | Documentation | Partial | High |
| Disaster recovery | DR workflow/documentation exists | Recovery objectives and restoration evidence need real execution | | RPO/RTO cannot be claimed as tested | High |
| Documentation | Extensive engineering documentation | Must continuously match implementation | | Some phase documents can outpace actual capability | High |

## Highest-priority blockers

### P0 — correctness/security

1. **FORCE RLS and local authentication context:** RLS applies to `users` and `user_roles`, while development registration/login did not originally establish the tenant setting before tenant-scoped writes/reads. This can break authentication or tempt future bypasses.
2. **Document parser error boundary:** malformed JSON and other decoding/parser errors can escape the API route instead of returning a controlled client error.
3. **Audit provenance:** persisted audit events currently omit the authenticated actor/request/IP in important paths, weakening evidentiary value.
4. **Frontend CSP correctness:** the current CSP is `default-src 'none'` without explicit script/style allowances while the frontend depends on a stylesheet and script; this needs verification/fix before claiming a secure usable web application.
5. **Latest CI:** the current `quality.yml` push run for `9346da5` is recorded as failed; success of the separate SBOM workflow is not evidence that the full quality gate is green.

### P1 — enterprise product depth

1. Concrete configurable data classification and policy model.
2. Findings/inventory first-class persistence and lifecycle.
3. Complete remediation lifecycle with evidence, ownership, deadlines, status transitions, and audit history.
4. Rich PIA data model and approval/reviewer controls.
5. Versioned enterprise resource APIs with pagination/filtering/idempotency/correlation IDs.
6. Concrete asynchronous ingestion/worker architecture with retries, idempotency and backpressure.
7. Connector adapter framework with at least one real provider implemented and tested.
8. Production observability: metrics, traces where justified, alerting, job health and security telemetry.

### P2 — acquisition evidence

1. Reproducible PII/NER benchmark datasets and per-class error analysis.
2. Independent penetration test and remediation evidence.
3. Security/privacy control evidence mapped to a recognized framework.
4. Tested backup/restore and disaster-recovery exercises with measured results.
5. Enterprise customer deployments, retention, expansion, and referenceable use cases.
6. Demonstrated integration value and measurable reduction in customer privacy/security workload.
7. Clear ownership/licensing/IP provenance for all code, models, datasets and dependencies.

## Acquisition thesis

The strategic wedge should be the combination of **sensitive-data discovery + explainable risk + regulatory mapping + governed remediation + tamper-evident evidence** rather than competing head-on with mature privacy suites on breadth.

A bank CTO would need evidence that the platform is not a prototype: repeatable security controls, tested tenant isolation, production identity integration, durable governance workflows, integration adapters, operational resilience, measurable detection quality, and real customer traction.

## Claims that must remain prohibited until evidence exists

- SOC 2, ISO 27001, PCI, HIPAA, GDPR/CCPA/Québec legal compliance certification claims.
- Government accreditation or regulatory approval.
- Production PII/ML accuracy percentages without a reproducible benchmark.
- Penetration-test claims without an actual independent test.
- Bank integrations without working provider adapters and integration tests.
- $9M–$15M valuation as a guaranteed outcome.

## Immediate execution order

1. Restore a genuinely green quality/E2E baseline.
2. Harden RLS/authentication and add database-backed cross-tenant tests.
3. Harden document parser/error handling and upload isolation boundaries.
4. Complete audit provenance and verification coverage.
5. Implement first-class classification + finding/inventory model.
6. Complete remediation and PIA APIs/workflows.
7. Implement concrete async worker/queue architecture.
8. Implement connector contracts and first production-grade adapter.
9. Expand observability and deployment/DR evidence.
10. Build the acquisition evidence package only from verified artifacts.
