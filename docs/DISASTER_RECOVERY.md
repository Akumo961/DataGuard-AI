# Disaster Recovery

Define RPO/RTO per deployment contract. Back up PostgreSQL on an approved schedule, encrypt backups, isolate backup credentials, retain according to policy, and regularly test restoration. Keep application images and migration artifacts versioned.

Recovery sequence: establish trusted infrastructure → restore secrets/identity integration → restore database → run/verify migrations → restore approved object data → start API/workers → health/readiness checks → validate tenant isolation → controlled traffic restoration.

A target government deployment must perform documented restore exercises and record evidence. This repository does not claim a tested RPO/RTO until measured in the target environment.
