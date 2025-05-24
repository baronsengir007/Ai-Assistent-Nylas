# pgvector Python Examples

This directory contains Python examples demonstrating how to use pgvector with PostgreSQL for vector similarity search.

## Prerequisites

1. Make sure the PostgreSQL container is running:
   ```bash
   cd ../docker
   docker compose up -d
   ```

2. Install Python dependencies:
   ```bash
   cd ..
   pip install -r requirements.txt
   ```

3. Copy the example environment file and add your OpenAI API key:
   ```bash
   cp ../env.example ../.env
   # Edit ../.env and add your OpenAI API key
   ```

## Examples

### 1. Basic Operations (`basic_operations.py`)

Demonstrates the fundamental operations with pgvector:
- Creating database connections
- Generating embeddings using OpenAI
- Inserting documents with embeddings
- Performing similarity searches

```bash
python basic_operations.py
```

### 2. SQLAlchemy Integration (`sqlalchemy_example.py`)

Shows how to use pgvector with SQLAlchemy ORM:
- Defining vector columns in SQLAlchemy models
- Creating a VectorStore class for document management
- Searching with metadata filters
- Managing documents with ORM patterns

```bash
python sqlalchemy_example.py
```

### 3. Hybrid Search (`hybrid_search.py`)

Implements hybrid search combining semantic and keyword search:
- Setting up full-text search alongside vector search
- Combining semantic similarity with keyword matching
- Adjusting weights between search methods
- Achieving better search results with hybrid approach

```bash
python hybrid_search.py
```

## Key Concepts

### Vector Operations

pgvector supports several distance functions:
- `<=>` - Cosine distance (default, good for normalized vectors)
- `<->` - L2/Euclidean distance
- `<#>` - Inner product (negative)

### Similarity vs Distance

When using cosine distance (`<=>`), convert to similarity:
```python
similarity = 1 - distance
```

### Index Types

- **IVFFlat**: Faster to build, good for most use cases
- **HNSW**: Better query performance, uses more memory

### Best Practices

1. **Batch Operations**: Generate embeddings in batches to reduce API calls
2. **Connection Pooling**: Use connection pools for production applications
3. **Index Tuning**: Adjust index parameters based on your dataset size
4. **Metadata Filtering**: Use GIN indexes on JSONB columns for efficient filtering

## Troubleshooting

If you encounter connection issues:
1. Check if the Docker container is running: `docker ps`
2. Verify the database URL in your `.env` file
3. Ensure pgvector extension is enabled: Connect to the database and run `\dx`

For embedding errors:
1. Verify your OpenAI API key is set correctly
2. Check your API quota and rate limits
3. Ensure you're using compatible embedding dimensions (1536 for text-embedding-3-small) 