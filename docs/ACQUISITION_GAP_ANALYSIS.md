# DataGuard — Bank-Grade Acquisition Technical Gap Analysis

**Baseline:** main `9346da5` plus the hardening branch through this assessment. This is an evidence-based engineering assessment; documentation is not proof and valuation is not guaranteed.

## Implementation matrix

| Capability | Status | Evidence / gap | Acquisition importance |
|---|---|---|---|
| Versioned API | 🟢 | `/api/v1` analysis, PIA, remediation, audit and findings routes | High |
| Authentication | 🟡 | JWT validation + development auth + OIDC/JWKS boundary; lifecycle/provider integration incomplete | Critical |
| Centralized RBAC | 🟢 | Central `AuthorizationPolicy`, fail-closed permissions | Critical |
| Tenant isolation | 🟢/🟡 | PostgreSQL RLS, tenant predicates and DB-level finding→analysis tenant FK; full resource mutation matrix still required | Critical |
| Secure uploads | 🟢/🟡 | Size, filename, magic/MIME, archive/OOXML and parser-error controls; malware isolation remains deployment work | Critical |
| PII detection | 🟢/🟡 | Regex/context + optional NER; independent benchmark absent | High |
| Classification | 🟢/🟡 | Explainable deterministic classifier with version provenance; organization policy configuration remains | Critical |
| Findings | 🟢/🟡 | Tenant-scoped persistence, RLS, API, classification evidence; full workflow lifecycle remains | Critical |
| Risk engine | 🟡 | Explainable deterministic scoring; policy calibration/evidence remain | High |
| Regulatory mapping | 🟡 | Versioned framework data/evaluator; evidence/applicability workflow remains basic | High |
| PIA | 🟡 | Persistent workflow/state machine; rich enterprise evidence/approval model remains | Critical |
| Remediation | 🟡 | Persistent items exist; SLA/evidence/approval lifecycle remains | Critical |
| Audit | 🟢/🟡 | Append-only hash chain + verification + external actor support; request/client provenance incomplete in some paths | Critical |
| Async processing | 🟡 | Redis Streams queue, retry/dead-letter contract and worker implementation added; production handler/storage integration not yet verified | Critical |
| Connectors | 🟡 | Provider-neutral connector contract added; no production provider adapter verified | Critical |
| Observability | 🟡 | Request IDs, logging/rate limits; metrics/traces/alerts need expansion | High |
| CI/CD | ⚠️ | Quality runs currently report failure with no retrievable jobs/check-runs via available API; security workflow added but not yet verified | Critical |
| Deployment | 🟡 | Docker/PostgreSQL/Redis foundation; Kubernetes/private-cloud/on-prem packaging incomplete | High |
| Disaster recovery | 🟡 | Architecture exists; restore exercises and measured RPO/RTO remain | High |
| Documentation | 🟡 | Security, threat, data-model, acquisition and due-diligence documentation expanded; consistency must track implementation | High |

## Remaining blockers

### P0

1. Obtain a real executable CI/E2E run and resolve its root failure.
2. Run full PostgreSQL tenant-isolation tests across read/create/update/delete/inference for every tenant-scoped resource.
3. Complete OIDC provisioning, role mapping, key rotation, session/revocation and provider acceptance tests.
4. Propagate request ID and client provenance through every auditable mutation.
5. Add malware scanning/quarantine/sandbox deployment controls for uploaded files.

### P1

1. Organization-configurable classification policies.
2. Findings lifecycle and evidence/approval workflow.
3. Full PIA data-subject/vendor/jurisdiction/safeguard/residual-risk model.
4. Remediation SLA, owner, evidence and verification workflow.
5. Durable job persistence/observability and a production-safe worker handler.
6. First real connector adapter with least-privilege integration tests.
7. Metrics, tracing, alerting and security-event operations.
8. Kubernetes/private-cloud deployment manifests and tested upgrade/rollback procedure.

### P2 — acquisition evidence

- Reproducible detection benchmark and error analysis.
- Independent penetration test.
- Assurance/control evidence.
- Tested backup/restore and measured RPO/RTO.
- Enterprise customers, retention and measurable ROI.
- IP/license provenance.

## Strategic wedge

The highest-potential wedge is **sensitive-data discovery + explainable classification/risk + regulatory mapping + governed remediation + tamper-evident evidence**. The product should not attempt to reproduce the full breadth of mature privacy suites before proving this wedge with regulated customers.

## Claims explicitly prohibited without evidence

No certification, regulatory approval, bank integration, independent security result, production detection accuracy, enterprise customer, revenue figure, proprietary-IP claim, or $9M–$15M valuation is asserted by this repository alone.
