# Phase 4 — Database & Multi-Tenancy

## Status

**Complete for the Phase 4 scope.** This phase establishes real PostgreSQL persistence, migrations, tenant-aware data models, persistent local authentication state, account lockout state, and database-enforced tenant isolation. It does not claim that the full application has been migrated to persistence; later phases own their domain workflows.

## Delivered

- SQLAlchemy 2.x async engine/session factory.
- PostgreSQL connection configured through application settings.
- Alembic migration environment and versioned schema.
- Organizations as first-class tenants.
- Users scoped to organizations with unique `(organization_id, email)` identity.
- Persistent roles scoped to organizations.
- API key metadata, analysis records, security events, audit events and refresh-token persistence.
- Failed-login counters and temporary account lockout state.
- Tenant-scoped query helper that fails closed when tenant context is absent.
- Transaction-local PostgreSQL tenant context.
- PostgreSQL Row-Level Security (RLS) enabled and forced for tenant-scoped tables.
- RLS policies fail closed when the transaction-local tenant setting is absent.
- Tenant foreign-key integrity for all tenant-scoped operational records.
- Database and Redis readiness checks.
- Schema and tenancy tests.

## Migration order

1. `20260826_0001_initial_enterprise_schema`
2. `20260826_0002_tenant_rls`
3. `20260826_0003_tenant_foreign_keys`

Run with:

```text
alembic upgrade head
```

## Tenant isolation contract

Application code must establish a tenant transaction using `tenant_transaction(session, organization_id)` before accessing RLS-protected tenant data. A missing tenant is rejected by the application helper and also rejected by PostgreSQL RLS.

The organization identifier is never accepted from an arbitrary request body as an authority. It must be derived from the authenticated principal/context in API layers introduced by later phases.

## Local development

Provide a PostgreSQL database and Redis instance matching the configured development URLs, install the project with `pip install -e '.[dev]'`, then run Alembic migrations before starting the API.

Production database credentials must come from a secret manager or deployment secret facility; repository defaults are development-only examples.

## Explicit non-claims

- No claim of database high availability.
- No claim of backup/restore validation; disaster recovery belongs to later deployment work.
- No claim of PostgreSQL encryption at rest; this is deployment/infrastructure dependent.
- No claim that every legacy prototype path has been converted to the new persistence layer.
- No certification, accreditation, government authorization or legal-compliance determination.
