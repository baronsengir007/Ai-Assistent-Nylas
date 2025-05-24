# PostgreSQL with pgvector Setup Guide

This guide provides a walkthrough for setting up a PostgreSQL database with the pgvector extension using Docker, and demonstrates how to interact with it using Python for vector similarity search operations.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Docker Setup](#docker-setup)
4. [Database Configuration](#database-configuration)
5. [Python Environment Setup](#python-environment-setup)
6. [Vector Operations Examples](#vector-operations-examples)
7. [Hybrid Search Implementation](#hybrid-search-implementation)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

pgvector is an open-source vector similarity search extension for PostgreSQL that enables:

- Storage and retrieval of high-dimensional vectors
- Exact and approximate nearest neighbor search
- Support for L2 distance, inner product, and cosine distance
- Integration with popular ML frameworks and embedding models

### Why pgvector?

- **Open Source**: Free to use with no licensing concerns
- **PostgreSQL Integration**: Leverage existing PostgreSQL features (ACID compliance, backups, replication)
- **Production Ready**: Used by many companies in production
- **Flexible**: Works with any embedding model and dimension size

## Prerequisites

- Docker Desktop installed
- Python 3.10+
- Basic knowledge of PostgreSQL and Python
- OpenAI API key (for embedding examples)

## Docker Setup

### Step 1: Create Docker Compose Configuration

Create `docker/docker-compose.yml`:

```yaml
services:
  pgvector:
    image: pgvector/pgvector:pg17
    container_name: pgvector-db
    restart: unless-stopped
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: postgres
    volumes:
      - pgvector_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pgvector_data:
```

### Step 2: Create Database Initialization Script

Create `docker/init.sql`:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a sample table for storing embeddings
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create a GIN index on metadata for efficient JSON queries
CREATE INDEX IF NOT EXISTS documents_metadata_idx ON documents 
USING GIN (metadata);
```

### Step 3: Start the PostgreSQL Container

```bash
cd docker
docker compose up -d
```

### Best Practices for Docker Setup

1. **Use pgvector Docker Image**: The `pgvector/pgvector` image comes with PostgreSQL and pgvector pre-installed
2. **Volume Persistence**: Always use named volumes for data persistence
3. **Health Checks**: Include health checks to ensure the database is ready
4. **Resource Limits**: For production, add resource limits:

```yaml
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

## Database Configuration

### Connection Details

```
Host: localhost
Port: 5432
Database: postgres
User: postgres
Password: postgres
```

### Verify pgvector Installation

use database gui exmaple here, i use table plus but give other options
Connect to the database and verify:

```bash
docker exec -it pgvector-db psql -U postgres -d postgres -c "\dx"
```

## Python Environment Setup

### Step 1: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

we use uv so you should just run uv sync in this project
Create `requirements.txt`:

```txt
psycopg[binary]==3.2.1
pgvector==0.3.2
openai==1.46.0
numpy==1.26.4
python-dotenv==1.0.1
sqlalchemy==2.0.30
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Environment Configuration

Create `.env`:

```env
# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here
```

## Vector Operations Examples

### Basic Vector Storage and Retrieval

### Using SQLAlchemy with pgvector

## Hybrid Search Implementation

## Best Practices

### 1. Index Selection

**IVFFlat Index** (Good for most use cases):
```sql
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```
- Faster build time
- Good recall with proper configuration
- `lists` = rows/1000 for < 1M rows

**HNSW Index** (Better recall, slower build):
```sql
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```
- Better query performance
- Higher memory usage
- Good for read-heavy workloads

### 2. Choosing Embedding Models

For production use:
- **OpenAI text-embedding-3-small**: Good balance of performance and cost
- **OpenAI text-embedding-3-large**: Better accuracy, higher cost
- **Open-source alternatives**: Consider models like BGE, E5, or Sentence Transformers

### 3. Connection Pooling

For production applications, use connection pooling:

```python
from psycopg_pool import ConnectionPool

pool = ConnectionPool(
    DATABASE_URL,
    min_size=5,
    max_size=20,
    timeout=30,
    max_idle=300
)
```

### 4. Batch Operations

For inserting multiple documents:

```python
def batch_insert_documents(conn, documents: List[Tuple[str, dict]]):
    embeddings = [get_embedding(content) for content, _ in documents]
    
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO documents (content, metadata, embedding)
            VALUES (%s, %s, %s)
        """, [
            (content, psycopg.types.json.Json(metadata), embedding)
            for (content, metadata), embedding in zip(documents, embeddings)
        ])
        conn.commit()
```

### 5. Self-Hosting Considerations

When deploying on a VM:
- **Memory**: Allocate sufficient RAM for indexes (check with `\di+` in psql)
- **Storage**: Use SSD for better performance
- **Backup**: Regular backups with `pg_dump`
- **Monitoring**: Use `pg_stat_statements` for query performance
- **Security**: Use SSL connections and strong passwords

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   # Check if container is running
   docker ps | grep pgvector-db
   
   # Check logs
   docker logs pgvector-db
   ```

2. **Extension Not Found**
   ```sql
   -- Make sure to enable the extension
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. **Dimension Mismatch**
   ```python
   # Ensure embedding dimensions match your model
   # OpenAI text-embedding-3-small: 1536 dimensions
   # OpenAI text-embedding-ada-002: 1536 dimensions
   ```

4. **Slow Queries**
   ```sql
   -- Check if index is being used
   EXPLAIN (ANALYZE, BUFFERS) 
   SELECT * FROM documents 
   ORDER BY embedding <=> '[...]'::vector 
   LIMIT 5;
   ```

### Performance Optimization

1. **Increase work_mem for index builds**:
   ```sql
   SET work_mem = '256MB';
   ```

2. **Vacuum regularly**:
   ```sql
   VACUUM ANALYZE documents;
   ```

3. **Monitor index usage**:
   ```sql
   SELECT schemaname, tablename, indexname, idx_scan
   FROM pg_stat_user_indexes
   WHERE tablename = 'documents';
   ```

## Additional Resources

- [pgvector GitHub Repository](https://github.com/pgvector/pgvector)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [pgvector Python Library Documentation](https://github.com/pgvector/pgvector-python)

## Next Steps

1. Explore different embedding models
2. Implement access control and security
3. Set up monitoring and alerting
4. Optimize indexes based on your query patterns
5. Build a complete RAG system using pgvector for vector storage
