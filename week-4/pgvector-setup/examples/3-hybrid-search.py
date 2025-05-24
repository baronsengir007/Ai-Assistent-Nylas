import os
import re
from typing import List

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector

load_dotenv()

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class HybridSearch:
    def __init__(self):
        self.conn = psycopg.connect(DATABASE_URL)
        register_vector(self.conn)
        self._setup_database()

    def _setup_database(self):
        """Set up database with full-text search capabilities."""
        with self.conn.cursor() as cur:
            # Add full-text search column if not exists
            cur.execute("""
                ALTER TABLE documents 
                ADD COLUMN IF NOT EXISTS search_vector tsvector
            """)

            # Create full-text search index
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_search_idx 
                ON documents USING GIN (search_vector)
            """)

            # Create trigger to update search_vector
            cur.execute("""
                CREATE OR REPLACE FUNCTION update_search_vector()
                RETURNS trigger AS $$
                BEGIN
                    NEW.search_vector := to_tsvector('english', NEW.content);
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                
                DROP TRIGGER IF EXISTS update_search_vector_trigger ON documents;
                
                CREATE TRIGGER update_search_vector_trigger
                BEFORE INSERT OR UPDATE ON documents
                FOR EACH ROW EXECUTE FUNCTION update_search_vector();
            """)

            # Update existing records
            cur.execute("""
                UPDATE documents 
                SET search_vector = to_tsvector('english', content)
                WHERE search_vector IS NULL
            """)

            self.conn.commit()

    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        response = client.embeddings.create(
            model="text-embedding-3-small", input=text, dimensions=1536
        )
        return response.data[0].embedding

    def hybrid_search(
        self, query: str, semantic_weight: float = 0.5, limit: int = 10
    ) -> List[dict]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query: Search query
            semantic_weight: Weight for semantic search (0-1)
            limit: Number of results to return
        """
        keyword_weight = 1 - semantic_weight

        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Prepare query for full-text search
        search_query = " & ".join(re.findall(r"\w+", query))

        with self.conn.cursor() as cur:
            # Hybrid search query
            cur.execute(
                """
                WITH semantic_search AS (
                    SELECT 
                        id,
                        1 - (embedding <=> %s::vector) as semantic_score
                    FROM documents
                    WHERE embedding IS NOT NULL
                ),
                keyword_search AS (
                    SELECT 
                        id,
                        ts_rank(search_vector, plainto_tsquery('english', %s)) as keyword_score
                    FROM documents
                    WHERE search_vector @@ plainto_tsquery('english', %s)
                )
                SELECT 
                    d.id,
                    d.content,
                    d.metadata,
                    COALESCE(s.semantic_score, 0) as semantic_score,
                    COALESCE(k.keyword_score, 0) as keyword_score,
                    (
                        COALESCE(s.semantic_score, 0) * %s + 
                        COALESCE(k.keyword_score, 0) * %s
                    ) as combined_score
                FROM documents d
                LEFT JOIN semantic_search s ON d.id = s.id
                LEFT JOIN keyword_search k ON d.id = k.id
                WHERE s.semantic_score IS NOT NULL OR k.keyword_score IS NOT NULL
                ORDER BY combined_score DESC
                LIMIT %s
            """,
                (
                    query_embedding,
                    search_query,
                    search_query,
                    semantic_weight,
                    keyword_weight,
                    limit,
                ),
            )

            results = cur.fetchall()

            return [
                {
                    "id": r[0],
                    "content": r[1],
                    "metadata": r[2],
                    "semantic_score": r[3],
                    "keyword_score": r[4],
                    "combined_score": r[5],
                }
                for r in results
            ]

    def add_document(self, content: str, metadata: dict = None):
        """Add a document to the database."""
        embedding = self.get_embedding(content)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (content, metadata, embedding)
                VALUES (%s, %s, %s)
                RETURNING id
            """,
                (content, psycopg.types.json.Json(metadata or {}), embedding),
            )

            doc_id = cur.fetchone()[0]
            self.conn.commit()
            return doc_id

    def close(self):
        """Close database connection."""
        self.conn.close()


def insert_documents():
    """Insert a list of documents into the database."""
    search_engine = HybridSearch()
    try:
        # Add sample documents
        documents = [
            {
                "content": "PostgreSQL is an advanced open-source relational database with powerful features for data management.",
                "metadata": {"category": "database", "type": "relational"},
            },
            {
                "content": "Vector databases enable similarity search by storing and querying high-dimensional embeddings.",
                "metadata": {"category": "database", "type": "vector"},
            },
            {
                "content": "Machine learning models transform text into numerical vectors for semantic analysis.",
                "metadata": {"category": "AI", "type": "embedding"},
            },
            {
                "content": "pgvector is a PostgreSQL extension that adds vector similarity search capabilities.",
                "metadata": {"category": "database", "type": "extension"},
            },
            {
                "content": "Hybrid search combines keyword matching with semantic similarity for better results.",
                "metadata": {"category": "search", "type": "hybrid"},
            },
        ]

        print("Adding documents...")
        for doc in documents:
            doc_id = search_engine.add_document(doc["content"], doc["metadata"])
            print(f"Added document {doc_id}")
    finally:
        search_engine.close()


def search_documents(query: str, semantic_weight: float = 0.5, limit: int = 3):
    """Search for documents using hybrid search."""
    search_engine = HybridSearch()
    try:
        print(f"\n--- Searching for: '{query}' ---")

        # Pure semantic search
        print("\n1. Pure Semantic Search (weight=1.0):")
        results = search_engine.hybrid_search(query, semantic_weight=1.0, limit=limit)
        for r in results:
            print(f"   Score: {r['combined_score']:.4f} - {r['content'][:60]}...")

        # Pure keyword search
        print("\n2. Pure Keyword Search (weight=0.0):")
        results = search_engine.hybrid_search(query, semantic_weight=0.0, limit=limit)
        for r in results:
            print(f"   Score: {r['combined_score']:.4f} - {r['content'][:60]}...")

        # Hybrid search
        print("\n3. Hybrid Search (weight=0.5):")
        results = search_engine.hybrid_search(query, semantic_weight=0.5, limit=limit)
        for r in results:
            print(
                f"   Semantic: {r['semantic_score']:.4f}, Keyword: {r['keyword_score']:.4f}"
            )
            print(f"   Combined: {r['combined_score']:.4f} - {r['content'][:60]}...")
    finally:
        search_engine.close()


def main():
    # Insert documents
    insert_documents()

    # Test different search strategies
    search_documents("pgvector is not a database")


if __name__ == "__main__":
    main()
