"""
Self-Query - RAG Optimization Technique #3

Self-query enables the retrieval system to extract structured filters from natural
language queries. This allows for more precise retrieval by combining semantic search
with metadata filtering, ensuring that results match both content and structural criteria.

Key Benefits:
- Combines semantic search with structured filtering
- Extracts metadata filters from natural language queries
- Improves precision by filtering on document attributes
- Handles complex queries with multiple constraints
"""

import sys
from pathlib import Path

# Add the rag-pipeline to the path so we can import from it
sys.path.append(str(Path(__file__).parent.parent) + "/rag-pipeline")

import json
from typing import Any, Dict, List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field
from rag.config import OPENAI_API_KEY  # type: ignore
from rag.rag_system import RAGSystem  # type: ignore


class MetadataSchema(BaseModel):
    """
    Pydantic model defining the available metadata fields and their possible values.
    """

    content_type: Optional[
        Literal["tutorial", "research", "documentation", "case_study"]
    ] = Field(default=None, description="Type of content")
    difficulty: Optional[Literal["beginner", "intermediate", "advanced"]] = Field(
        default=None, description="Content difficulty level"
    )
    topic: Optional[
        Literal["rag", "embeddings", "llm", "vector_search", "ai_applications"]
    ] = Field(default=None, description="Main topic area")


class QueryComponents(BaseModel):
    """
    Pydantic model for structured output of query component extraction.
    """

    semantic_query: str = Field(
        description="The main semantic content query, cleaned of metadata-specific terms"
    )
    filters: Optional[MetadataSchema] = Field(
        default=None,
        description="Metadata filters extracted from the query, or null if no filters apply",
    )


class SelfQuery:
    """
    Implements Self-Query technique for RAG systems with synthetic data.
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.rag_system = RAGSystem()

    def setup_synthetic_data(self):
        """Clear database and insert synthetic documents about RAG/AI topics."""
        print("Clearing existing documents...")
        self.rag_system.vector_store.clear_all_documents()

        print("Inserting synthetic RAG/AI documents...")

        # Documents designed to show self-query benefits
        documents = [
            # Beginner RAG tutorials
            {
                "content": "Introduction to RAG: Retrieval-Augmented Generation combines the power of large language models with external knowledge retrieval. This beginner-friendly guide explains the basic concepts and architecture.",
                "metadata": {
                    "content_type": "tutorial",
                    "difficulty": "beginner",
                    "topic": "rag",
                },
            },
            {
                "content": "Getting started with vector embeddings: Learn how to convert text into numerical vectors for semantic search. This tutorial covers the basics of embedding models and similarity calculations.",
                "metadata": {
                    "content_type": "tutorial",
                    "difficulty": "beginner",
                    "topic": "embeddings",
                },
            },
            # Intermediate RAG content
            {
                "content": "Advanced RAG techniques: Explore sophisticated methods like query expansion, re-ranking, and hybrid search to improve retrieval quality in production systems.",
                "metadata": {
                    "content_type": "tutorial",
                    "difficulty": "intermediate",
                    "topic": "rag",
                },
            },
            {
                "content": "Vector database optimization: Performance tuning strategies for large-scale vector search including indexing methods, query optimization, and memory management.",
                "metadata": {
                    "content_type": "documentation",
                    "difficulty": "intermediate",
                    "topic": "vector_search",
                },
            },
            # Advanced research content
            {
                "content": "Research findings on RAG performance: Our study evaluated different retrieval strategies across 10,000 queries, showing that hybrid approaches outperform pure semantic search by 23%.",
                "metadata": {
                    "content_type": "research",
                    "difficulty": "advanced",
                    "topic": "rag",
                },
            },
            {
                "content": "Advanced embedding techniques: Novel approaches to fine-tuning embedding models for domain-specific applications, including contrastive learning and multi-task training.",
                "metadata": {
                    "content_type": "research",
                    "difficulty": "advanced",
                    "topic": "embeddings",
                },
            },
            # LLM content
            {
                "content": "Large Language Model integration patterns: Best practices for incorporating LLMs into production applications, covering prompt engineering, context management, and response validation.",
                "metadata": {
                    "content_type": "documentation",
                    "difficulty": "intermediate",
                    "topic": "llm",
                },
            },
            {
                "content": "LLM prompt optimization for beginners: Simple techniques to improve your prompts including clear instructions, examples, and structured outputs.",
                "metadata": {
                    "content_type": "tutorial",
                    "difficulty": "beginner",
                    "topic": "llm",
                },
            },
            # AI Applications case studies
            {
                "content": "Case study: Building a customer support AI system using RAG. This real-world example shows how we reduced response time by 60% while maintaining 95% accuracy.",
                "metadata": {
                    "content_type": "case_study",
                    "difficulty": "intermediate",
                    "topic": "ai_applications",
                },
            },
            {
                "content": "AI-powered document analysis: Advanced techniques for extracting insights from unstructured documents using multi-modal models and semantic understanding.",
                "metadata": {
                    "content_type": "case_study",
                    "difficulty": "advanced",
                    "topic": "ai_applications",
                },
            },
            # More diverse content to show filtering benefits
            {
                "content": "Vector search fundamentals: Understanding similarity metrics, indexing algorithms like HNSW and IVF, and choosing the right approach for your use case.",
                "metadata": {
                    "content_type": "tutorial",
                    "difficulty": "beginner",
                    "topic": "vector_search",
                },
            },
            {
                "content": "Production RAG system architecture: Designing scalable retrieval systems with proper caching, load balancing, and monitoring for enterprise applications.",
                "metadata": {
                    "content_type": "documentation",
                    "difficulty": "advanced",
                    "topic": "rag",
                },
            },
        ]

        for doc in documents:
            # Create embedding for the content
            embedding = self.rag_system.embedding_service.create_embedding(
                doc["content"]
            )

            # Add document to vector store
            self.rag_system.vector_store.add_document(
                content=doc["content"], embedding=embedding, metadata=doc["metadata"]
            )

        print(f"Inserted {len(documents)} documents")

    def extract_query_components(self, query: str) -> Dict[str, Any]:
        """Extract semantic query and metadata filters from natural language query using structured output."""
        # Get the JSON schema for the metadata
        metadata_schema_json = MetadataSchema.model_json_schema()

        try:
            response = self.client.responses.parse(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "system",
                        "content": f"""You are an expert at analyzing search queries and extracting structured information.

                        Given a user's natural language query, extract:
                        1. The main semantic query (what they're looking for content-wise, cleaned of metadata terms)
                        2. Any metadata filters that can be applied

                        Available metadata schema:
                        {json.dumps(metadata_schema_json, indent=2)}

                        Only include filters if they are clearly implied by the query. If no filters apply, set filters to null.
                        The semantic query should focus on the core content need, removing metadata-specific terms.""",
                    },
                    {"role": "user", "content": f"User query: '{query}'"},
                ],
                text_format=QueryComponents,
                temperature=0.0,
            )

            # Extract the parsed response
            parsed_result = response.output_parsed

            # Convert the Pydantic model to a dictionary for filters
            filters_dict = {}
            if parsed_result.filters:
                filters_dict = parsed_result.filters.model_dump(exclude_none=True)

            return {
                "semantic_query": parsed_result.semantic_query,
                "filters": filters_dict,
            }

        except Exception as e:
            print(f"Error extracting query components: {e}")
            return {"semantic_query": query, "filters": {}}

    def apply_filters(
        self, documents: List[Dict], filters: Dict[str, Any]
    ) -> List[Dict]:
        """Apply metadata filters to documents."""
        if not filters:
            return documents

        filtered = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            if all(metadata.get(k) == v for k, v in filters.items()):
                filtered.append(doc)

        return filtered

    def search_with_self_query(self, query: str, k: int = 8) -> Dict[str, Any]:
        """Perform self-query search and return detailed results."""
        print(f"🔍 Query: '{query}'")

        # Extract components
        components = self.extract_query_components(query)
        semantic_query = components["semantic_query"]
        filters = components["filters"]

        print(f"📝 Semantic query: '{semantic_query}'")
        print(f"🏷️ Filters: {filters}")

        # Retrieve documents
        retrieved_docs = self.rag_system.retrieve_context(semantic_query, k=k)
        print(f"📚 Retrieved {len(retrieved_docs)} documents")

        # Apply filters
        if filters:
            filtered_docs = self.apply_filters(retrieved_docs, filters)
            print(f"🔍 After filtering: {len(filtered_docs)} documents")
            final_docs = filtered_docs
        else:
            print("🔍 No filters applied")
            final_docs = retrieved_docs

        return {
            "query": query,
            "semantic_query": semantic_query,
            "filters": filters,
            "retrieved_count": len(retrieved_docs),
            "filtered_count": len(final_docs),
            "documents": final_docs[:5],  # Top 5 for display
        }

    def compare_search_methods(self, query: str):
        """Compare standard vs self-query search."""
        print(f"\n{'=' * 60}")
        print(f"COMPARISON: {query}")
        print("=" * 60)

        # Standard search
        print("\n1️⃣ STANDARD SEARCH:")
        print("-" * 30)
        standard_docs = self.rag_system.retrieve_context(query, k=5)
        print(f"Retrieved: {len(standard_docs)} documents")
        for i, doc in enumerate(standard_docs, 1):
            metadata = doc.get("metadata", {})
            print(
                f"  {i}. [{metadata.get('content_type', '?')}|{metadata.get('difficulty', '?')}|{metadata.get('topic', '?')}] {doc['content'][:60]}..."
            )

        # Self-query search
        print("\n2️⃣ SELF-QUERY SEARCH:")
        print("-" * 30)
        self_query_result = self.search_with_self_query(query, k=8)

        print("\nFiltered documents:")
        for i, doc in enumerate(self_query_result["documents"], 1):
            metadata = doc.get("metadata", {})
            print(
                f"  {i}. [{metadata.get('content_type', '?')}|{metadata.get('difficulty', '?')}|{metadata.get('topic', '?')}] {doc['content'][:60]}..."
            )

        # Show the benefit
        improvement = self_query_result["filtered_count"] - len(standard_docs)
        if improvement > 0:
            print(f"\n✅ Self-query found {improvement} more relevant documents!")
        elif self_query_result["filters"]:
            print("\n🎯 Self-query provided more precise results through filtering!")
        else:
            print("\n📊 Both methods returned similar results (no filters applied)")


def demonstrate_self_query():
    """Demonstrate self-query with focused examples."""
    print("🧠 Self-Query Demonstration")
    print("=" * 50)

    # Initialize and setup data
    self_query = SelfQuery()
    self_query.setup_synthetic_data()

    # Test queries designed to show self-query benefits
    test_queries = [
        "Show me beginner tutorials about RAG systems",
        "Find advanced research on embeddings",
        "I need documentation for vector search",
    ]

    print(f"\n🧪 Testing {len(test_queries)} queries that benefit from self-query")

    for query in test_queries:
        self_query.compare_search_methods(query)


if __name__ == "__main__":
    demonstrate_self_query()
