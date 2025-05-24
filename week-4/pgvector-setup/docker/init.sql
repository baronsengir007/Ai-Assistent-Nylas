-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a sample table for storing embeddings
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSON,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
-- Note on IVFFlat index configuration:
-- - For small datasets (< 1000 vectors): Use 2-10 lists
-- - For medium datasets (1000-10000 vectors): Use sqrt(n) lists
-- - For large datasets (> 10000 vectors): Use n/1000 lists
-- 
-- Current setting (lists = 2) is optimized for this demo with 2 documents.
-- In production, adjust this number based on your dataset size:
--   - Too few lists: Slower searches but more accurate
--   - Too many lists: Faster but might miss relevant matches
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 2);

-- Create a GIN index on metadata for efficient JSON queries
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents 
USING GIN (metadata); 