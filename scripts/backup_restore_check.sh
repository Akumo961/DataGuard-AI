#!/usr/bin/env bash
set -euo pipefail

# Reproducible PostgreSQL backup/restore exercise for CI and operator runbooks.
# The caller must already have the DataGuard PostgreSQL service running.

backup_file="${1:-/tmp/dataguard-backup.dump}"
service="${DATAGUARD_DB_SERVICE:-postgres}"

docker compose exec -T "$service" pg_dump \
  --format=custom \
  --no-owner \
  --no-privileges \
  --file=/tmp/dataguard-backup.dump \
  dataguard

docker compose cp "$service:/tmp/dataguard-backup.dump" "$backup_file"
test -s "$backup_file"

# Restore into an isolated database so the live CI database remains untouched.
docker compose exec -T "$service" psql -v ON_ERROR_STOP=1 -U dataguard -d postgres \
  -c 'DROP DATABASE IF EXISTS dataguard_restore_check;'
docker compose exec -T "$service" psql -v ON_ERROR_STOP=1 -U dataguard -d postgres \
  -c 'CREATE DATABASE dataguard_restore_check;'
docker compose cp "$backup_file" "$service:/tmp/dataguard-restore.dump"
docker compose exec -T "$service" pg_restore \
  --exit-on-error \
  --no-owner \
  --no-privileges \
  --dbname=dataguard_restore_check \
  /tmp/dataguard-restore.dump

count="$(docker compose exec -T "$service" psql -At -U dataguard -d dataguard_restore_check -c 'SELECT count(*) FROM organizations;')"
test "$count" -ge 1
printf 'Backup/restore verified: restored organizations=%s\n' "$count"

docker compose exec -T "$service" psql -v ON_ERROR_STOP=1 -U dataguard -d postgres \
  -c 'DROP DATABASE dataguard_restore_check;'
