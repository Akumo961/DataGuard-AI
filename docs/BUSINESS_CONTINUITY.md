# Business Continuity

Business continuity planning should identify critical services, dependencies, owners, alternate operating procedures and recovery priorities. Maintain documented contacts and escalation paths outside the application.

Critical dependencies include identity, PostgreSQL, object storage, Redis/job infrastructure, document-processing workers, monitoring and DNS/ingress. Define service priorities and manual fallback procedures with the contracting organization.

Continuity objectives must be agreed and tested per deployment; this repository does not claim contractual availability or RTO/RPO guarantees.
