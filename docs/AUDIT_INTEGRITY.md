# Audit integrity

DataGuard audit events are designed as an append-only, tenant-scoped hash chain.

Each event stores:

- `previous_hash`: the integrity hash of the preceding event for the tenant;
- `integrity_hash`: SHA-256 over a canonical serialization of the event fields and `previous_hash`.

PostgreSQL writers for the same tenant are serialized with a transaction-scoped advisory lock so concurrent inserts cannot silently fork the chain. The application ORM rejects updates and deletes of `AuditEvent` rows.

The `dataguard.audit.integrity.verify_chain()` function validates both the stored hash and predecessor relationship. Tests cover middle-record tampering and reordering.

This is **tamper-evident**, not an absolute immutability guarantee against a database administrator or compromised database superuser. Production high-assurance deployments should additionally ship audit events to an independent write-once/WORM or external security-information-and-event-management store and periodically anchor chain heads outside the primary database.
