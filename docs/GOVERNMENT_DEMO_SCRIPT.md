# Government Demonstration — 15 Minutes

## Safety

Use only the synthetic dataset in `demo/synthetic/`. It is deliberately marked as demonstration data and uses `.invalid` addresses. Never import real personal information into the demo environment.

## 0:00–1:00 — Executive context

Explain the problem: organizations need visibility into where personal information exists, how sensitive it is, what risks require attention, and what evidence supports governance decisions. Clarify that DataGuard is decision support and does not certify legal compliance.

## 1:00–3:00 — Data discovery

Open the dashboard and show the discovery/analysis entry point. Select a synthetic document and demonstrate validation before processing.

## 3:00–5:00 — PII detection

Run analysis. Show detected categories, confidence and redacted values. Explain that deterministic patterns and layered detection are used rather than presenting an LLM as authoritative.

## 5:00–7:00 — AI-assisted classification and risk

Show classification/risk output and contributing factors. Emphasize explainability and human review rather than an automatic legal decision.

## 7:00–9:00 — Québec privacy controls

Open the compliance area and demonstrate configurable controls/evidence requirements. Explain that legal applicability and current references require qualified review.

## 9:00–11:00 — PIA

Create a PIA in DRAFT, enter system, purpose, information, sources, recipients, storage, retention, risks and safeguards, then move it through review/remediation.

## 11:00–12:30 — Remediation

Assign remediation ownership, record the action and return the PIA to review. Demonstrate that workflow transitions are controlled and versioned.

## 12:30–14:00 — Audit/evidence

Show audit history and evidence references. Explain integrity hashing and why production deployments additionally require durable/immutable evidence infrastructure.

## 14:00–15:00 — Executive conclusion

Show risk posture and report outputs. Close with the implementation roadmap: enterprise identity, connectors, production persistence, deployment controls, security validation and organizational/legal governance.

## Buyer questions to anticipate

- Where is data processed and stored?
- Which identity provider is supported?
- Can findings be reviewed by humans?
- How are false positives handled?
- How is tenant isolation validated?
- What evidence is retained?
- Which controls are implemented versus planned?
- What must be configured for Québec/Canadian residency and sovereignty requirements?
