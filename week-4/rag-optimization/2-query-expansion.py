"""
Query Expansion - RAG Optimization Technique #2

Query expansion improves retrieval by generating multiple related queries from the
original user query. This increases recall by capturing documents that might use
different terminology or phrasing than the original query.

Key Benefits:
- Increases recall by covering more semantic variations
- Handles vocabulary mismatch between query and documents
- Improves retrieval for ambiguous or short queries
- Works especially well with keyword-based search (BM25)
"""

import sys
from pathlib import Path

# Add the rag-pipeline to the path so we can import from it
sys.path.append(str(Path(__file__).parent.parent) + "/rag-pipeline")

from openai import OpenAI
from rag.rag_system import RAGSystem  # type: ignore
from rag.config import OPENAI_API_KEY  # type: ignore
from typing import List, Dict, Any
from pydantic import BaseModel, Field


class ExpandedQueries(BaseModel):
    """
    Pydantic model for structured output of expanded queries.
    """

    expanded_queries: List[str] = Field(
        description="List of alternative queries that rephrase the original question using different terminology, structure, or context"
    )


class QueryExpansion:
    """
    Implements Query Expansion technique for improved RAG retrieval.

    This class generates multiple related queries from a single user query,
    then retrieves documents for each expanded query and combines the results.
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.rag_system = RAGSystem()

    def expand_query(self, original_query: str, num_expansions: int = 4) -> List[str]:
        """
        Generate multiple related queries from the original query using structured output.

        Args:
            original_query: The user's original question
            num_expansions: Number of additional queries to generate

        Returns:
            List of expanded queries including the original
        """
        try:
            response = self.client.responses.parse(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "system",
                        "content": f"""You are an expert at generating search queries. Given a user's question, 
                        generate {num_expansions} alternative ways to ask the same question that might 
                        help retrieve relevant documents. The alternatives should:

                        1. Use different terminology or synonyms
                        2. Rephrase the question structure
                        3. Break down complex queries into simpler parts
                        4. Add context that might be implied

                        Generate exactly {num_expansions} alternative queries that maintain the same intent but use different wording.""",
                    },
                    {"role": "user", "content": f"Original question: {original_query}"},
                ],
                text_format=ExpandedQueries,
            )

            # Extract the expanded queries from the structured response
            expanded_queries = response.output_parsed.expanded_queries

            # Ensure we have the right number of queries and include the original
            expanded_queries = expanded_queries[:num_expansions]
            all_queries = [original_query] + expanded_queries

            return all_queries

        except Exception as e:
            print(f"Error expanding query: {e}")
            return [original_query]  # Fallback to original query only

    def retrieve_with_expanded_queries(
        self, queries: List[str], k_per_query: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents for each expanded query and combine results.

        This is the core of query expansion: instead of searching once, we search
        multiple times with different phrasings, then deduplicate and combine results.
        This increases coverage by finding documents that match different terminology.

        Args:
            queries: List of queries to search with (original + expanded)
            k_per_query: Number of documents to retrieve per query

        Returns:
            Combined and deduplicated list of retrieved documents
        """
        all_results = []
        seen_content = set()  # For deduplication - tracks content we've already seen

        # Search the vector database with each expanded query
        for i, query in enumerate(queries):
            print(f"Retrieving for query {i + 1}: '{query}'")

            # Each query searches the entire vector database independently
            results = self.rag_system.retrieve_context(query, k=k_per_query)

            # Process each document from this query's results
            for doc in results:
                # Use content hash for deduplication (first 200 chars to identify unique docs)
                content_hash = hash(doc["content"][:200])

                # Only add if we haven't seen this document content before
                if content_hash not in seen_content:
                    seen_content.add(content_hash)

                    # Add metadata about which query retrieved this document
                    doc["metadata"] = doc.get("metadata", {})
                    doc["metadata"]["retrieved_by_query"] = query
                    doc["metadata"]["query_index"] = i
                    doc["metadata"]["is_original_query"] = i == 0

                    all_results.append(doc)

        # Sort all unique results by similarity score (highest first)
        all_results.sort(key=lambda x: x["similarity"], reverse=True)

        print(
            f"📊 Combined {len(queries)} searches into {len(all_results)} unique documents"
        )
        return all_results

    def query_with_expansion(
        self,
        question: str,
        num_expansions: int = 4,
        k_per_query: int = 3,
        final_k: int = 5,
        show_expansion_details: bool = False,
    ) -> str:
        """
        Perform query expansion and retrieve documents.

        Args:
            question: Original user question
            num_expansions: Number of query expansions to generate
            k_per_query: Documents to retrieve per expanded query
            final_k: Final number of documents to use for generation
            show_expansion_details: Whether to show expansion process details

        Returns:
            Generated response using expanded query results
        """
        # Step 1: Expand the query
        print("🔍 Expanding query...")
        expanded_queries = self.expand_query(question, num_expansions)

        if show_expansion_details:
            print(f"\nOriginal query: '{question}'")
            print(f"Expanded to {len(expanded_queries)} queries:")
            for i, query in enumerate(expanded_queries):
                print(f"  {i + 1}. {query}")
            print()

        # Step 2: Retrieve with expanded queries
        print("📚 Retrieving documents with expanded queries...")
        all_results = self.retrieve_with_expanded_queries(expanded_queries, k_per_query)

        if show_expansion_details:
            print(f"\nRetrieved {len(all_results)} unique documents:")
            for i, doc in enumerate(all_results[:final_k]):
                query_info = doc["metadata"].get("retrieved_by_query", "Unknown")
                is_original = doc["metadata"].get("is_original_query", False)
                marker = "🎯" if is_original else "🔄"
                print(
                    f"  {marker} Doc {i + 1} (sim: {doc['similarity']:.3f}): Retrieved by '{query_info[:50]}...'"
                )
            print()

        # Step 3: Use top-k documents for generation
        final_context = all_results[:final_k]

        # Generate response
        print("🤖 Generating response...")
        response = self.rag_system.generate_response(question, final_context)

        return response

    def compare_with_without_expansion(self, question: str) -> Dict[str, Any]:
        """
        Compare results with and without query expansion.

        Returns:
            Dictionary containing both results for comparison
        """
        print("🔬 Comparing Query Expansion vs Standard Retrieval")
        print("=" * 60)

        # Standard retrieval (without expansion)
        print("\n1️⃣ Standard Retrieval (No Expansion):")
        print("-" * 40)
        standard_context = self.rag_system.retrieve_context(question, k=5)
        standard_response = self.rag_system.generate_response(
            question, standard_context
        )

        print(f"Retrieved {len(standard_context)} documents")
        for i, doc in enumerate(standard_context):
            print(f"  Doc {i + 1}: Similarity {doc['similarity']:.3f}")

        # Query expansion retrieval
        print("\n2️⃣ Query Expansion Retrieval:")
        print("-" * 40)
        expanded_response = self.query_with_expansion(
            question,
            num_expansions=4,
            k_per_query=3,
            final_k=5,
            show_expansion_details=True,
        )

        return {
            "question": question,
            "standard_response": standard_response,
            "expanded_response": expanded_response,
            "standard_context": standard_context,
        }


def demonstrate_query_expansion():
    """
    Demonstrate the query expansion technique with practical examples.
    """
    print("🔍 Query Expansion Demonstration")
    print("=" * 50)

    # Initialize query expansion system
    query_expander = QueryExpansion()

    # Make sure we have documents in the vector store
    # (Assuming documents were already ingested from previous examples)
    doc_count = query_expander.rag_system.vector_store.get_document_count()
    if doc_count == 0:
        print("No documents found in vector store. Ingesting sample document...")
        from rag.config import DOCLING_PAPER_URL  # type: ignore

        query_expander.rag_system.ingest_document(DOCLING_PAPER_URL)

    print(
        f"Vector store contains {query_expander.rag_system.vector_store.get_document_count()} documents"
    )

    # Test queries that benefit from expansion
    test_queries = [
        "How does document parsing work?",  # Could be expanded to include "text extraction", "OCR", etc.
        "What are the benefits of this system?",  # Vague query that needs expansion
    ]

    print(f"\n🧪 Testing Query Expansion with Sample Queries")
    print("=" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"TEST {i}: {query}")
        print("=" * 60)

        # Show comparison between standard and expanded retrieval
        comparison = query_expander.compare_with_without_expansion(query)

        print(f"\n📊 RESULTS COMPARISON:")
        print("-" * 30)
        print(f"Standard Response: {comparison['standard_response'][:200]}...")
        print(f"\nExpanded Response: {comparison['expanded_response'][:200]}...")

        print(f"\n" + "=" * 60)


if __name__ == "__main__":
    demonstrate_query_expansion()
