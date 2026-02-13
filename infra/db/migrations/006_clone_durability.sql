-- Bloom Database Migration 006: Durable resumable clone pipeline
-- Adds checkpoint columns, heartbeat, cancel support, and orphan detection

-- ============================================
-- New checkpoint columns on clone_jobs
-- ============================================

-- Sub-stage tracking within L1/L2
ALTER TABLE clone_jobs ADD COLUMN IF NOT EXISTS substage VARCHAR(50);

-- Retry tracking
ALTER TABLE clone_jobs ADD COLUMN IF NOT EXISTS retry_count INT NOT NULL DEFAULT 0;

-- L1 per-generator checkpoint (each saved independently)
ALTER TABLE clone_jobs ADD COLUMN IF NOT EXISTS l1_topics JSONB;
ALTER TABLE clone_jobs ADD COLUMN IF NOT EXISTS l1_shades JSONB;
ALTER TABLE clone_jobs ADD COLUMN IF NOT EXISTS l1_bio JSONB;

-- L2 per-type completion tracking: {"qa": true, "identity": true, ...}
ALTER TABLE clone_jobs ADD COLUMN IF NOT EXISTS l2_completed_types JSONB NOT NULL DEFAULT '{}'::jsonb;

-- Heartbeat for orphan detection
ALTER TABLE clone_jobs ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ;

-- Cancellation flag
ALTER TABLE clone_jobs ADD COLUMN IF NOT EXISTS cancelled BOOLEAN NOT NULL DEFAULT FALSE;

-- ============================================
-- Update stage CHECK constraint to include L1 sub-stages
-- ============================================

-- Drop old constraint and add expanded one
ALTER TABLE clone_jobs DROP CONSTRAINT IF EXISTS clone_jobs_stage_check;
ALTER TABLE clone_jobs ADD CONSTRAINT clone_jobs_stage_check CHECK (
    stage IN (
        'pending',
        'l1_topics',
        'l1_shades',
        'l1_bio',
        'l1_profile',
        'l2_synthesis',
        'l2_merge',
        'training',
        'exporting',
        'completed',
        'failed'
    )
);

-- ============================================
-- Index for orphan detection query
-- ============================================

CREATE INDEX IF NOT EXISTS idx_clone_jobs_orphan_detection
    ON clone_jobs (stage, heartbeat_at)
    WHERE stage NOT IN ('completed', 'failed');
