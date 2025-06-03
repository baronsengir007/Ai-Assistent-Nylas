"""
Search Pipelines Module for RAG Metrics Tutorial

This module contains the search pipeline implementations that are used
for evaluation experiments. It's extracted from the setup file to allow
for easy importing in the experiments module.
"""

import sys
from pathlib import Path

# Add required paths
sys.path.append(str(Path(__file__).parent.parent) + "/pgvector-setup")
sys.path.append(str(Path(__file__).parent.parent) + "/rag-optimization")

import os
import time
from typing import List, Dict, Any, Literal
import psycopg
from pgvector.psycopg import register_vector
from openai import OpenAI
import cohere
from dotenv import load_dotenv

load_dotenv()

# Configuration
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))


class SearchPipeline:
    """Base class for search pipelines."""

    def __init__(self, name: str):
        self.name = name
        self.conn = psycopg.connect(DATABASE_URL)
        register_vector(self.conn)
        self.embedding_dimensions = 1536

    def get_embedding(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI."""
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=self.embedding_dimensions,
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return [0.0] * self.embedding_dimensions

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError

    def close(self):
        """Close database connection."""
        self.conn.close()


class SemanticSearchPipeline(SearchPipeline):
    """Pure semantic search using vector similarity."""

    def __init__(self):
        super().__init__("semantic_search")

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity."""
        query_embedding = self.get_embedding(query)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    id,
                    content,
                    metadata,
                    1 - (embedding <=> %s::vector) as similarity,
                    (embedding <#> %s::vector) * -1 as inner_product_score
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, query_embedding, k),
            )

            results = cur.fetchall()

            return [
                {
                    "id": r[0],
                    "content": r[1],
                    "metadata": r[2],
                    "similarity": float(r[3]),
                    "inner_product_score": float(r[4]),
                    "method": self.name,
                    "rank": i + 1,
                }
                for i, r in enumerate(results)
            ]


class HybridSearchPipeline(SearchPipeline):
    """Hybrid search combining semantic and keyword search using RRF."""

    def __init__(self):
        super().__init__("hybrid_search")
        self.default_rrf_k = 50
        self.search_config = "english"

    def search(
        self,
        query: str,
        k: int = 10,
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
        keyword_mode: Literal["strict", "flexible"] = "flexible",
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search."""

        # Handle keyword search mode
        if keyword_mode == "strict":
            tsquery_func = (
                f"websearch_to_tsquery('{self.search_config}', %(fts_query)s)"
            )
            fts_query = query
        else:  # flexible
            tsquery_func = f"to_tsquery('{self.search_config}', %(fts_query)s)"
            # Convert to OR query
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT websearch_to_tsquery('{self.search_config}', %s)::text",
                    (query,),
                )
                result = cur.fetchone()[0]
                fts_query = (
                    result.replace(" & ", " | ") if result and result != "''" else "''"
                )

        query_embedding = self.get_embedding(query)

        with self.conn.cursor() as cur:
            sql_query = f"""
                WITH full_text AS (
                    SELECT 
                        id,
                        ts_rank_cd(fts, {tsquery_func}) as fts_score,
                        row_number() OVER (
                            ORDER BY 
                                ts_rank_cd(fts, {tsquery_func}) DESC,
                                length(content)
                        ) as rank_ix
                    FROM documents
                    WHERE fts @@ {tsquery_func}
                    ORDER BY rank_ix
                    LIMIT %(k)s * 2
                ),
                semantic AS (
                    SELECT 
                        id,
                        1 - (embedding <=> %(query_embedding)s::vector) as similarity,
                        row_number() OVER (
                            ORDER BY embedding <=> %(query_embedding)s::vector
                        ) as rank_ix
                    FROM documents
                    ORDER BY rank_ix
                    LIMIT %(k)s * 2
                )
                SELECT 
                    d.*,
                    ft.fts_score,
                    s.similarity,
                    COALESCE(1.0 / (%(rrf_k)s + ft.rank_ix), 0.0) * %(keyword_weight)s as keyword_score,
                    COALESCE(1.0 / (%(rrf_k)s + s.rank_ix), 0.0) * %(semantic_weight)s as semantic_score,
                    (
                        COALESCE(1.0 / (%(rrf_k)s + ft.rank_ix), 0.0) * %(keyword_weight)s + 
                        COALESCE(1.0 / (%(rrf_k)s + s.rank_ix), 0.0) * %(semantic_weight)s
                    ) as combined_score
                FROM 
                    full_text ft
                    FULL OUTER JOIN semantic s ON ft.id = s.id
                    JOIN documents d ON COALESCE(ft.id, s.id) = d.id
                ORDER BY combined_score DESC
                LIMIT %(k)s
            """

            params = {
                "fts_query": fts_query,
                "query_embedding": query_embedding,
                "k": k,
                "rrf_k": self.default_rrf_k,
                "keyword_weight": keyword_weight,
                "semantic_weight": semantic_weight,
            }

            cur.execute(sql_query, params)
            results = cur.fetchall()

            return [
                {
                    "id": r[0],
                    "content": r[1],
                    "metadata": r[2],
                    "embedding": r[3],
                    "fts": r[4],
                    "created_at": r[5],
                    "fts_score": float(r[6]) if r[6] else 0.0,
                    "similarity": float(r[7]) if r[7] else 0.0,
                    "keyword_score": float(r[8]),
                    "semantic_score": float(r[9]),
                    "combined_score": float(r[10]),
                    "method": self.name,
                    "rank": i + 1,
                }
                for i, r in enumerate(results)
            ]


class HybridRerankingPipeline(SearchPipeline):
    """Hybrid search with Cohere reranking."""

    def __init__(self):
        super().__init__("hybrid_reranking")
        self.hybrid_search = HybridSearchPipeline()
        self.rerank_model = "rerank-english-v3.0"

    def search(
        self,
        query: str,
        k: int = 10,
        initial_k: int = None,
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search followed by Cohere reranking."""

        # Get more documents initially for reranking
        initial_k = initial_k or min(k * 3, 50)  # Get 3x more documents for reranking

        # Get initial results from hybrid search
        initial_results = self.hybrid_search.search(
            query=query,
            k=initial_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
        )

        if not initial_results:
            return []

        # Prepare documents for reranking
        doc_texts = [doc["content"] for doc in initial_results]

        try:
            # Rerank using Cohere
            rerank_response = cohere_client.rerank(
                model=self.rerank_model,
                query=query,
                documents=doc_texts,
                top_n=k,
                return_documents=True,
            )

            # Process reranked results
            reranked_docs = []
            for i, result in enumerate(rerank_response.results):
                original_doc = initial_results[result.index].copy()
                original_doc.update(
                    {
                        "rerank_score": result.relevance_score,
                        "original_rank": result.index + 1,
                        "final_rank": i + 1,
                        "method": self.name,
                        "rank": i + 1,
                    }
                )
                reranked_docs.append(original_doc)

            return reranked_docs

        except Exception as e:
            print(f"❌ Error during reranking: {e}")
            # Fallback to original hybrid results
            return initial_results[:k]

    def close(self):
        """Close database connections."""
        super().close()
        self.hybrid_search.close()


class SearchPipelineManager:
    """Manages all search pipelines and provides unified interface."""

    def __init__(self):
        """Initialize all search pipelines."""
        self.pipelines = {
            "semantic": SemanticSearchPipeline(),
            "hybrid": HybridSearchPipeline(),
            "hybrid_reranking": HybridRerankingPipeline(),
        }

    def get_pipeline(self, name: str) -> SearchPipeline:
        """Get a specific search pipeline by name."""
        if name not in self.pipelines:
            raise ValueError(
                f"Unknown pipeline: {name}. Available: {list(self.pipelines.keys())}"
            )
        return self.pipelines[name]

    def search_all(self, query: str, k: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Run the same query across all pipelines."""
        results = {}

        print(f"🔍 Running query across all pipelines: '{query[:50]}...'")

        for name, pipeline in self.pipelines.items():
            print(f"  Running {name}...")
            start_time = time.time()

            try:
                pipeline_results = pipeline.search(query, k=k)
                execution_time = (time.time() - start_time) * 1000  # Convert to ms

                # Add execution time to each result
                for result in pipeline_results:
                    result["execution_time_ms"] = execution_time

                results[name] = pipeline_results
                print(
                    f"    ✅ {len(pipeline_results)} results in {execution_time:.1f}ms"
                )

            except Exception as e:
                print(f"    ❌ Error: {e}")
                results[name] = []

        return results

    def test_pipelines(self, test_queries: List[str] = None):
        """Test all pipelines with sample queries."""
        if test_queries is None:
            test_queries = [
                "What is machine learning?",
                "How does artificial intelligence work?",
                "Explain deep learning algorithms",
            ]

        print("🧪 Testing Search Pipelines")
        print("=" * 50)

        for i, query in enumerate(test_queries, 1):
            print(f"\nTest Query {i}: {query}")
            print("-" * 30)

            results = self.search_all(query, k=5)

            # Show top result from each pipeline
            for pipeline_name, pipeline_results in results.items():
                if pipeline_results:
                    top_result = pipeline_results[0]
                    score_info = self._format_score_info(top_result)
                    print(
                        f"{pipeline_name:15} | {score_info} | {top_result['content'][:60]}..."
                    )
                else:
                    print(f"{pipeline_name:15} | No results")

        print("\n✅ Pipeline testing complete!")

    def _format_score_info(self, result: Dict[str, Any]) -> str:
        """Format score information for display."""
        if "combined_score" in result:
            return f"Score: {result['combined_score']:.4f}"
        elif "rerank_score" in result:
            return f"Rerank: {result['rerank_score']:.4f}"
        elif "similarity" in result:
            return f"Sim: {result['similarity']:.4f}"
        else:
            return "Score: N/A"

    def get_pipeline_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all pipelines."""
        return {
            "semantic": {
                "name": "Semantic Search",
                "description": "Pure vector similarity search using cosine similarity",
                "features": [
                    "Vector embeddings",
                    "Semantic understanding",
                    "Fast retrieval",
                ],
            },
            "hybrid": {
                "name": "Hybrid Search",
                "description": "Combines semantic and keyword search using Reciprocal Rank Fusion",
                "features": [
                    "Vector + keyword search",
                    "RRF combination",
                    "Balanced relevance",
                ],
            },
            "hybrid_reranking": {
                "name": "Hybrid + Reranking",
                "description": "Hybrid search with Cohere reranking for improved precision",
                "features": ["Hybrid search", "Cohere reranking", "High precision"],
            },
        }

    def close_all(self):
        """Close all pipeline connections."""
        for pipeline in self.pipelines.values():
            pipeline.close()
