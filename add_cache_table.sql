-- Add processing cache table to database
CREATE TABLE IF NOT EXISTS processing_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key VARCHAR(64) UNIQUE NOT NULL,
    result_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_processing_cache_key ON processing_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_processing_cache_created ON processing_cache(created_at);