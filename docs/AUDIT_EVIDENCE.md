# Audit & Evidence Center

Phase 12 introduces the domain and service foundation for auditable actions and evidence generation.

## Audit records

Records capture timestamp, user, organization, action, object, previous state, new state, IP address where appropriate, request ID, result and an integrity hash. Hash verification supports tamper-evidence; it is not a cryptographic signature or proof of immutability by itself.

The persistence layer must enforce append-only semantics, restricted deletion, retention policy and access controls in production. Database-backed immutable/WORM storage can be introduced without changing the domain contract.

## Evidence

`EvidenceItem` carries organization ownership, evidence type, title, object reference, framework/version metadata, structured content, creator and timestamp. This supports privacy, risk, PIA, remediation and control evidence without requiring raw sensitive document contents to be copied into audit records.

## Privacy principle

Audit/evidence payloads should contain the minimum information needed to establish accountability. Raw PII must not be logged. Production integrations should redact secrets and sensitive values before persistence.

## Chain integrity

The service accepts the preceding record hash so a persistence implementation can build a per-tenant append-only hash chain. Verification detects modification or chain discontinuity; operational controls are still required to protect the underlying store and keys.
