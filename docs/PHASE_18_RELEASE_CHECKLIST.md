# Phase 18 Release Checklist

## Engineering

- [ ] CI green on release commit
- [ ] Unit/integration/API/security suites green
- [ ] No unresolved critical/high defects
- [ ] Dependency and container scans reviewed
- [ ] SBOM generated and retained
- [ ] Release artifact provenance/signing configured

## Security

- [ ] OIDC/JWKS configured and tested
- [ ] Privileged access management validated
- [ ] TLS and network policy validated
- [ ] Secrets/KMS configuration validated
- [ ] Upload malware scanning and isolated workers enabled
- [ ] Penetration test completed

## Privacy/governance

- [ ] Tenant isolation adversarial testing completed
- [ ] Retention/deletion policies configured
- [ ] Audit/evidence retention validated
- [ ] Québec/Canadian control mappings reviewed by qualified personnel
- [ ] PIA workflow accepted by privacy stakeholders

## AI/ML

- [ ] Approved evaluation dataset available
- [ ] Precision/recall/F1 and per-class results reproducible
- [ ] False-positive/false-negative review completed
- [ ] Model provenance/versioning documented
- [ ] Human review process operational

## Operations

- [ ] Monitoring and alerting configured
- [ ] Backup restore test passed
- [ ] Disaster recovery exercise passed
- [ ] Capacity/load test passed
- [ ] Incident response tested
- [ ] Support/escalation process approved

## Product

- [ ] Accessibility/UAT accepted
- [ ] Government demo verified with synthetic data
- [ ] User/admin documentation approved
- [ ] Procurement/commercial material reviewed
