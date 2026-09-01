# Disaster Recovery

## Scope

PostgreSQL is the system of record. Redis is a cache/queue component and is not required to reconstruct persisted analyses or governance records.

## Recovery objectives

These are operational targets, not guarantees until validated in the target deployment:

- **RPO:** 24 hours with daily logical backups; reduce this target with managed PostgreSQL point-in-time recovery in production.
- **RTO:** 4 hours for a standard single-region database recovery.
- Backups must be encrypted, access-controlled, retained according to the organization's retention policy, and stored outside the primary failure domain.

## Backup

Use PostgreSQL `pg_dump` in custom format for portable logical backups. Record backup timestamp, database version, schema migration level and checksum. Do not place production backup files in Git.

## Restore procedure

1. Provision an isolated PostgreSQL instance.
2. Restore the latest known-good backup with `pg_restore --exit-on-error`.
3. Apply any migrations required by the target release.
4. Verify organization, analysis, PIA, remediation and audit tables.
5. Verify audit-chain continuity before returning the service to users.
6. Record the drill/recovery evidence and elapsed recovery time.

## Automated recovery drill

`.github/workflows/dr-drill.yml` runs weekly and on demand against synthetic data. It creates the schema, inserts a synthetic organization marker, takes a custom-format backup, restores it into an isolated database and verifies critical tables/data.

A successful CI drill demonstrates that the documented procedure is executable; it does not replace provider-level backups, cross-region recovery or an operational disaster exercise.
