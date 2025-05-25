import os
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
            # Create HNSW index for vector search if not exists
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents USING hnsw (embedding vector_ip_ops)
            """)

            # Create GIN index for full-text search if not exists
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_fts_idx 
                ON documents USING GIN(fts)
            """)

            self.conn.commit()

    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        response = client.embeddings.create(
            model="text-embedding-3-small", input=text, dimensions=1536
        )
        return response.data[0].embedding

    def hybrid_search(
        self,
        query: str,
        semantic_weight: float = 1.0,
        full_text_weight: float = 1.0,
        rrf_k: int = 50,
        limit: int = 10,
    ) -> List[dict]:
        """
        Perform hybrid search combining semantic and keyword search using Reciprocal Rank Fusion (RRF).

        Source: https://supabase.com/docs/guides/ai/hybrid-search

        Args:
            query: Search query
            semantic_weight: Weight for semantic search
            full_text_weight: Weight for full-text search
            rrf_k: Smoothing constant for RRF
            limit: Number of results to return
        """
        # Get query embedding
        query_embedding = self.get_embedding(query)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                WITH full_text AS (
                    SELECT 
                        id,
                        row_number() OVER (
                            ORDER BY 
                                ts_rank_cd(fts, plainto_tsquery('english', %s)) DESC,
                                length(content)
                        ) as rank_ix
                    FROM documents
                    WHERE fts @@ plainto_tsquery('english', %s)
                    ORDER BY rank_ix
                    LIMIT least(%s, 30) * 2
                ),
                semantic AS (
                    SELECT 
                        id,
                        row_number() OVER (ORDER BY embedding <#> %s::vector) as rank_ix
                    FROM documents
                    ORDER BY rank_ix
                    LIMIT least(%s, 30) * 2
                )
                SELECT 
                    d.*,
                    COALESCE(1.0 / (%s + ft.rank_ix), 0.0) * %s as full_text_score,
                    COALESCE(1.0 / (%s + s.rank_ix), 0.0) * %s as semantic_score,
                    (
                        COALESCE(1.0 / (%s + ft.rank_ix), 0.0) * %s + 
                        COALESCE(1.0 / (%s + s.rank_ix), 0.0) * %s
                    ) as combined_score
                FROM documents d
                FULL OUTER JOIN full_text ft ON d.id = ft.id
                FULL OUTER JOIN semantic s ON d.id = s.id
                WHERE ft.id IS NOT NULL OR s.id IS NOT NULL
                ORDER BY combined_score DESC
                LIMIT least(%s, 30)
                """,
                (
                    query,
                    query,
                    limit,
                    query_embedding,
                    limit,
                    rrf_k,
                    full_text_weight,
                    rrf_k,
                    semantic_weight,
                    rrf_k,
                    full_text_weight,
                    rrf_k,
                    semantic_weight,
                    limit,
                ),
            )

            results = cur.fetchall()

            return [
                {
                    "id": r[0],
                    "content": r[1],
                    "metadata": r[2],
                    "embedding": r[3],
                    "fts": r[4],
                    "created_at": r[5],
                    "full_text_score": float(r[6]),
                    "semantic_score": float(r[7]),
                    "combined_score": float(r[8]),
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

    def debug_conversational_query(self):
        """Debug exactly what happens with the conversational query."""

        query = "How can I implement vector similarity search in my PostgreSQL database? I want to find similar documents based on their content."

        print(f"=== DETAILED DEBUG: Conversational Query ===")
        print(f"Query: {query}")
        print()

        with self.conn.cursor() as cur:
            # 1. See how plainto_tsquery processes this query
            cur.execute("SELECT plainto_tsquery('english', %s)", (query,))
            parsed_query = cur.fetchone()[0]
            print(f"plainto_tsquery result: {parsed_query}")
            print()

            # 2. Check each document individually
            cur.execute("SELECT id, content, fts FROM documents ORDER BY id")
            documents = cur.fetchall()

            for doc_id, content, fts in documents:
                print(f"--- Document {doc_id} ---")
                print(f"Content: {content}")
                print(f"FTS tokens: {fts}")

                # Test if this specific document matches
                cur.execute(
                    """
                    SELECT 
                        %s::tsvector @@ plainto_tsquery('english', %s) as matches,
                        ts_rank_cd(%s::tsvector, plainto_tsquery('english', %s)) as score
                """,
                    (fts, query, fts, query),
                )

                matches, score = cur.fetchone()
                print(f"Matches: {matches}")
                print(f"Score: {score}")
                print()

            # 3. Test with simpler queries to see what works
            test_queries = [
                "vector",
                "similarity",
                "search",
                "postgresql",
                "implement",
                "vector similarity",
                "postgresql vector",
                "implement vector search",
            ]

            print("=== Testing simpler queries ===")
            for test_query in test_queries:
                cur.execute(
                    """
                    SELECT COUNT(*) 
                    FROM documents 
                    WHERE fts @@ plainto_tsquery('english', %s)
                """,
                    (test_query,),
                )

                match_count = cur.fetchone()[0]
                print(f"'{test_query}': {match_count} matches")

            print()

            # 4. Show what terms from the conversational query actually exist in our documents
            print("=== Token analysis ===")

            # Get all unique tokens from our documents
            cur.execute("""
                SELECT DISTINCT unnest(tsvector_to_array(fts)) as token
                FROM documents
                ORDER BY token
            """)

            all_tokens = [row[0] for row in cur.fetchall()]
            print(f"All tokens in our documents: {all_tokens}")
            print()

            # See what the conversational query becomes as tokens
            # We'll parse the tsquery string manually since we can't cast tsquery to tsvector
            cur.execute("SELECT plainto_tsquery('english', %s)::text", (query,))
            query_string = cur.fetchone()[0]

            # Extract tokens from the tsquery string
            import re

            if query_string and query_string != "''":
                query_tokens = re.findall(r"'([^']+)'", query_string)
            else:
                query_tokens = []

            print(f"Query string: {query_string}")
            print(f"Query tokens extracted: {query_tokens}")
            print()

            # Find overlap
            overlapping_tokens = set(all_tokens) & set(query_tokens)
            print(f"Overlapping tokens: {overlapping_tokens}")

            if not overlapping_tokens:
                print("❌ NO OVERLAPPING TOKENS! This explains the 0.0000 scores.")
            else:
                print(f"✅ Found {len(overlapping_tokens)} overlapping tokens")

    def simple_fts_test(self):
        """Simple test to verify FTS is working at all."""

        print("=== SIMPLE FTS TEST ===")

        with self.conn.cursor() as cur:
            # Test 1: Simple word that should definitely match
            print("Test 1: Search for 'vector' (should match multiple docs)")
            cur.execute("""
                SELECT id, content, 
                    fts @@ plainto_tsquery('english', 'vector') as matches,
                    ts_rank_cd(fts, plainto_tsquery('english', 'vector')) as score
                FROM documents 
                WHERE fts @@ plainto_tsquery('english', 'vector')
                ORDER BY score DESC
            """)

            results = cur.fetchall()
            if results:
                for doc_id, content, matches, score in results:
                    print(f"  ✓ Doc {doc_id}: Score {score:.6f} - {content[:50]}...")
            else:
                print("  ❌ No matches for 'vector' - something is very wrong!")
            print()

            # Test 2: Test the exact tokens we see in your database
            print("Test 2: Search for exact tokens from your database")
            exact_tokens = ["postgresql", "vector", "search", "similar"]

            for token in exact_tokens:
                cur.execute(
                    """
                    SELECT COUNT(*) 
                    FROM documents 
                    WHERE fts @@ plainto_tsquery('english', %s)
                """,
                    (token,),
                )

                count = cur.fetchone()[0]
                print(f"  '{token}': {count} matches")
            print()

            # Test 3: What does plainto_tsquery actually produce for our problem query?
            problem_query = "How can I implement vector similarity search in my PostgreSQL database?"

            cur.execute("SELECT plainto_tsquery('english', %s)::text", (problem_query,))
            query_result = cur.fetchone()[0]
            print(f"Test 3: plainto_tsquery for problem query")
            print(f"  Input: {problem_query}")
            print(f"  Output: {query_result}")

            # Test 4: Manual token matching
            print(
                f"\nTest 4: Manual check - do any documents contain the query tokens?"
            )

            # Extract the individual terms from the tsquery
            if query_result and query_result != "''":
                # Parse the tsquery result to get individual terms
                import re

                terms = re.findall(r"'([^']+)'", query_result)
                print(f"  Individual terms: {terms}")

                for term in terms:
                    cur.execute(
                        """
                        SELECT id, content
                        FROM documents 
                        WHERE fts @@ to_tsquery('english', %s)
                        LIMIT 1
                    """,
                        (term,),
                    )

                    result = cur.fetchone()
                    if result:
                        doc_id, content = result
                        print(f"    '{term}' matches Doc {doc_id}: {content[:30]}...")
                    else:
                        print(f"    '{term}' matches: NONE")
            else:
                print("  ❌ plainto_tsquery returned empty result!")

    def build_smart_or_query(self, query: str) -> str:
        """
        Build a smart OR query that only includes terms that exist in our documents.
        """
        with self.conn.cursor() as cur:
            # Get all available tokens from our documents
            cur.execute("""
                SELECT DISTINCT unnest(tsvector_to_array(fts)) as token
                FROM documents
            """)
            available_tokens = set(row[0] for row in cur.fetchall())

            # Get what plainto_tsquery would extract (but we'll use for OR instead of AND)
            cur.execute("SELECT plainto_tsquery('english', %s)::text", (query,))
            query_string = cur.fetchone()[0]

            # Extract tokens from the query
            import re

            if query_string and query_string != "''":
                query_tokens = re.findall(r"'([^']+)'", query_string)
            else:
                # Fallback: basic word extraction
                words = query.lower().split()
                query_tokens = [word.strip(".,?!") for word in words if len(word) > 2]

            # Only keep tokens that exist in our documents
            matching_tokens = [
                token for token in query_tokens if token in available_tokens
            ]

            # Remove duplicates while preserving order
            unique_tokens = []
            seen = set()
            for token in matching_tokens:
                if token not in seen:
                    seen.add(token)
                    unique_tokens.append(token)

            if not unique_tokens:
                # Fallback to a basic search
                return "vector | search"

            # Create OR query
            or_query = " | ".join(unique_tokens)
            return or_query

    def smart_hybrid_search(
        self,
        query: str,
        semantic_weight: float = 1.0,
        full_text_weight: float = 1.0,
        rrf_k: int = 50,
        limit: int = 10,
    ) -> List[dict]:
        """
        Smart hybrid search that builds OR queries from available tokens.
        """

        # Build a smart OR query
        or_query = self.build_smart_or_query(query)
        print(f"Original query: {query}")
        print(f"Smart OR query: {or_query}")

        # Get query embedding
        query_embedding = self.get_embedding(query)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                WITH full_text AS (
                    SELECT 
                        id,
                        ts_rank_cd(fts, to_tsquery('english', %s)) as raw_score,
                        row_number() OVER (
                            ORDER BY 
                                ts_rank_cd(fts, to_tsquery('english', %s)) DESC,
                                length(content)
                        ) as rank_ix
                    FROM documents
                    WHERE fts @@ to_tsquery('english', %s)
                    ORDER BY rank_ix
                    LIMIT least(%s, 30) * 2
                ),
                semantic AS (
                    SELECT 
                        id,
                        row_number() OVER (ORDER BY embedding <#> %s::vector) as rank_ix
                    FROM documents
                    ORDER BY rank_ix
                    LIMIT least(%s, 30) * 2
                )
                SELECT 
                    d.*,
                    ft.raw_score as fts_raw_score,
                    COALESCE(1.0 / (%s + ft.rank_ix), 0.0) * %s as full_text_score,
                    COALESCE(1.0 / (%s + s.rank_ix), 0.0) * %s as semantic_score,
                    (
                        COALESCE(1.0 / (%s + ft.rank_ix), 0.0) * %s + 
                        COALESCE(1.0 / (%s + s.rank_ix), 0.0) * %s
                    ) as combined_score
                FROM documents d
                FULL OUTER JOIN full_text ft ON d.id = ft.id
                FULL OUTER JOIN semantic s ON d.id = s.id
                WHERE ft.id IS NOT NULL OR s.id IS NOT NULL
                ORDER BY combined_score DESC
                LIMIT least(%s, 30)
                """,
                (
                    or_query,  # to_tsquery with OR logic
                    or_query,  # for ORDER BY
                    or_query,  # for WHERE clause
                    limit,
                    query_embedding,
                    limit,
                    rrf_k,
                    full_text_weight,
                    rrf_k,
                    semantic_weight,
                    rrf_k,
                    full_text_weight,
                    rrf_k,
                    semantic_weight,
                    limit,
                ),
            )

            results = cur.fetchall()

            return [
                {
                    "id": r[0],
                    "content": r[1],
                    "metadata": r[2],
                    "embedding": r[3],
                    "fts": r[4],
                    "created_at": r[5],
                    "fts_raw_score": float(r[6]) if r[6] else 0.0,
                    "full_text_score": float(r[7]),
                    "semantic_score": float(r[8]),
                    "combined_score": float(r[9]),
                }
                for r in results
            ]

    def test_smart_search(self):
        """Test the smart search approach."""

        query = (
            "How can I implement vector similarity search in my PostgreSQL database?"
        )

        print("=== TESTING SMART HYBRID SEARCH ===")

        # Test pure FTS with smart OR query
        print("\n1. Pure Full-Text Search (smart OR):")
        results = self.smart_hybrid_search(
            query, semantic_weight=0.0, full_text_weight=1.0, limit=3
        )

        for r in results:
            print(
                f"   FTS Score: {r['fts_raw_score']:.4f} | Combined: {r['combined_score']:.4f}"
            )
            print(f"   Content: {r['content'][:60]}...")

        print("\n2. Hybrid Search (smart OR + semantic):")
        results = self.smart_hybrid_search(
            query, semantic_weight=1.0, full_text_weight=1.0, limit=3
        )

        for r in results:
            print(
                f"   FTS: {r['full_text_score']:.4f}, Semantic: {r['semantic_score']:.4f}"
            )
            print(f"   Combined: {r['combined_score']:.4f} - {r['content'][:60]}...")


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
                "content": "To implement vector similarity search, you need to store document embeddings and compute distances.",
                "metadata": {"category": "implementation", "type": "tutorial"},
            },
            {
                "content": "Machine learning models transform text content into numerical vectors for semantic analysis.",
                "metadata": {"category": "AI", "type": "embedding"},
            },
            {
                "content": "pgvector is a PostgreSQL extension that adds vector similarity search capabilities to your database.",
                "metadata": {"category": "database", "type": "extension"},
            },
            {
                "content": "Hybrid search combines keyword matching with semantic similarity for better document retrieval results.",
                "metadata": {"category": "search", "type": "hybrid"},
            },
        ]

        print("Adding documents...")
        for doc in documents:
            doc_id = search_engine.add_document(doc["content"], doc["metadata"])
            print(f"Inserted document {doc_id}: {doc['content'][:50]}...")
    finally:
        search_engine.close()


def search_documents(
    semantic_weight: float = 1.0,
    full_text_weight: float = 1.0,
    limit: int = 3,
):
    """Search for documents using hybrid search."""
    search_engine = HybridSearch()
    try:
        # Test Case 1: Conversational Query
        print("\n=== Testing with a conversational query ===")
        conversational_query = "How can I implement vector similarity search in my PostgreSQL database? I want to find similar documents based on their content."
        conversational_query = "PostgreSQL"
        print(f"\n--- Searching for: '{conversational_query}' ---")

        # Pure semantic search
        print("\n1. Pure Semantic Search (semantic_weight=1.0, full_text_weight=0.0):")
        results = search_engine.hybrid_search(
            conversational_query, semantic_weight=1.0, full_text_weight=0.0, limit=limit
        )
        for r in results:
            print(f"   Score: {r['combined_score']:.4f} - {r['content'][:60]}...")

        # Pure keyword search
        print("\n2. Pure Keyword Search (semantic_weight=0.0, full_text_weight=1.0):")
        results = search_engine.hybrid_search(
            conversational_query, semantic_weight=0.0, full_text_weight=1.0, limit=limit
        )
        for r in results:
            print(f"   Score: {r['combined_score']:.4f} - {r['content'][:60]}...")

        # Hybrid search
        print("\n3. Hybrid Search (semantic_weight=1.0, full_text_weight=1.0):")
        results = search_engine.hybrid_search(
            conversational_query, semantic_weight=1.0, full_text_weight=1.0, limit=limit
        )
        for r in results:
            print(
                f"   Full Text: {r['full_text_score']:.4f}, Semantic: {r['semantic_score']:.4f}"
            )
            print(f"   Combined: {r['combined_score']:.4f} - {r['content'][:60]}...")

        # Test Case 2: Technical Query
        print("\n=== Testing with a technical query ===")
        technical_query = "pgvector PostgreSQL extension"
        print(f"\n--- Searching for: '{technical_query}' ---")

        # Pure semantic search
        print("\n1. Pure Semantic Search (semantic_weight=1.0, full_text_weight=0.0):")
        results = search_engine.hybrid_search(
            technical_query, semantic_weight=1.0, full_text_weight=0.0, limit=limit
        )
        for r in results:
            print(f"   Score: {r['combined_score']:.4f} - {r['content'][:60]}...")

        # Pure keyword search
        print("\n2. Pure Keyword Search (semantic_weight=0.0, full_text_weight=1.0):")
        results = search_engine.hybrid_search(
            technical_query, semantic_weight=0.0, full_text_weight=1.0, limit=limit
        )
        for r in results:
            print(f"   Score: {r['combined_score']:.4f} - {r['content'][:60]}...")

        # Hybrid search
        print("\n3. Hybrid Search (semantic_weight=1.0, full_text_weight=1.0):")
        results = search_engine.hybrid_search(
            technical_query, semantic_weight=1.0, full_text_weight=1.0, limit=limit
        )
        for r in results:
            print(
                f"   Full Text: {r['full_text_score']:.4f}, Semantic: {r['semantic_score']:.4f}"
            )
            print(f"   Combined: {r['combined_score']:.4f} - {r['content'][:60]}...")

        # Test Case 3: Adjusted Weights
        print("\n=== Testing with adjusted weights ===")
        print(f"\n--- Searching for: '{conversational_query}' ---")
        print("Using semantic_weight=1.5, full_text_weight=0.5")

        # Hybrid search with adjusted weights
        print("\nHybrid Search (semantic_weight=1.5, full_text_weight=0.5):")
        results = search_engine.hybrid_search(
            conversational_query, semantic_weight=1.5, full_text_weight=0.5, limit=limit
        )
        for r in results:
            print(
                f"   Full Text: {r['full_text_score']:.4f}, Semantic: {r['semantic_score']:.4f}"
            )
            print(f"   Combined: {r['combined_score']:.4f} - {r['content'][:60]}...")

    finally:
        search_engine.close()


# def main():
#     # Insert documents
#     # insert_documents()

#     # Test different search strategies
#     search_documents()


def main():
    search_engine = HybridSearch()
    try:
        search_engine.test_smart_search()
    finally:
        search_engine.close()


if __name__ == "__main__":
    main()
