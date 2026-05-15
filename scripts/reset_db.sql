-- Honcho database reset script
-- Drops all application data while preserving extensions and schema structure.
-- Run with: psql $DATABASE_URL -f scripts/reset_db.sql
--       or: docker exec -i honcho-database-1 psql -U postgres -d postgres -f /dev/stdin < scripts/reset_db.sql

BEGIN;

-- Disable FK checks during truncation
SET session_replication_role = replica;

TRUNCATE TABLE
    public.queue,
    public.active_queue_sessions,
    public.webhook_endpoints,
    public.perception_events,
    public.message_embeddings,
    public.messages,
    public.session_peers,
    public.documents,
    public.collections,
    public.sessions,
    public.peers,
    public.workspaces
CASCADE;

-- Re-enable FK checks
SET session_replication_role = DEFAULT;

COMMIT;

SELECT 'Honcho database reset complete.' AS status;
