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

from openai import OpenAI
from rag.rag_system import RAGSystem  # type: ignore
from rag.config import OPENAI_API_KEY  # type: ignore
from typing import List, Dict, Any
import json
import re


class SelfQuery:
    """
    Implements Self-Query technique for RAG systems.

    This class extracts structured metadata filters from natural language queries,
    enabling more precise retrieval by combining semantic search with filtering.
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.rag_system = RAGSystem()

        # Define available metadata fields for filtering
        self.metadata_schema = {
            "document_type": {
                "type": "string",
                "description": "Type of document (e.g., 'research_paper', 'technical_doc', 'tutorial')",
                "values": [
                    "research_paper",
                    "technical_doc",
                    "tutorial",
                    "documentation",
                ],
            },
            "section": {
                "type": "string",
                "description": "Document section (e.g., 'introduction', 'methodology', 'results', 'conclusion')",
                "values": [
                    "introduction",
                    "methodology",
                    "results",
                    "conclusion",
                    "abstract",
                    "references",
                ],
            },
            "topic": {
                "type": "string",
                "description": "Main topic or subject area",
                "values": [
                    "machine_learning",
                    "document_processing",
                    "nlp",
                    "computer_vision",
                    "data_science",
                ],
            },
            "difficulty": {
                "type": "string",
                "description": "Content difficulty level",
                "values": ["beginner", "intermediate", "advanced"],
            },
            "page_number": {
                "type": "integer",
                "description": "Page number in the document",
            },
        }

    def extract_query_components(self, query: str) -> Dict[str, Any]:
        """
        Extract semantic query and metadata filters from natural language query.

        Args:
            query: Natural language query from user

        Returns:
            Dictionary with 'semantic_query' and 'filters'
        """
        # Create schema description for the prompt
        schema_desc = ""
        for field, info in self.metadata_schema.items():
            schema_desc += f"- {field} ({info['type']}): {info['description']}\n"
            if "values" in info:
                schema_desc += f"  Possible values: {', '.join(info['values'])}\n"

        prompt = f"""
        You are an expert at analyzing search queries and extracting structured information.
        
        Given a user's natural language query, extract:
        1. The main semantic query (what they're looking for content-wise)
        2. Any metadata filters that can be applied
        
        Available metadata fields:
        {schema_desc}
        
        User query: "{query}"
        
        Respond with a JSON object containing:
        {{
            "semantic_query": "the main content query",
            "filters": {{
                "field_name": "value",
                ...
            }}
        }}
        
        Only include filters if they are clearly implied by the query. If no filters apply, use an empty object for filters.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )

            # Parse the JSON response
            response_text = response.choices[0].message.content.strip()

            # Extract JSON from response (handle cases where there might be extra text)
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)

                # Validate the result structure
                if "semantic_query" not in result:
                    result["semantic_query"] = query
                if "filters" not in result:
                    result["filters"] = {}

                return result
            else:
                # Fallback if JSON parsing fails
                return {"semantic_query": query, "filters": {}}

        except Exception as e:
            print(f"Error extracting query components: {e}")
            return {"semantic_query": query, "filters": {}}

    def apply_metadata_filter(
        self, documents: List[Dict[str, Any]], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply metadata filters to retrieved documents.

        Args:
            documents: List of retrieved documents
            filters: Dictionary of metadata filters to apply

        Returns:
            Filtered list of documents
        """
        if not filters:
            return documents

        filtered_docs = []

        for doc in documents:
            doc_metadata = doc.get("metadata", {})
            matches_all_filters = True

            for filter_key, filter_value in filters.items():
                if filter_key in doc_metadata:
                    doc_value = doc_metadata[filter_key]

                    # Handle different comparison types
                    if isinstance(filter_value, str):
                        if doc_value.lower() != filter_value.lower():
                            matches_all_filters = False
                            break
                    elif isinstance(filter_value, (int, float)):
                        if doc_value != filter_value:
                            matches_all_filters = False
                            break
                else:
                    # If the document doesn't have the required metadata field
                    matches_all_filters = False
                    break

            if matches_all_filters:
                filtered_docs.append(doc)

        return filtered_docs

    def enrich_documents_with_metadata(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add sample metadata to documents for demonstration purposes.
        In a real system, this metadata would come from the document processing pipeline.
        """
        enriched_docs = []

        for i, doc in enumerate(documents):
            # Create sample metadata based on content analysis
            content = doc["content"].lower()

            # Determine document type
            if any(
                word in content
                for word in ["abstract", "methodology", "results", "conclusion"]
            ):
                doc_type = "research_paper"
            elif any(word in content for word in ["tutorial", "guide", "how to"]):
                doc_type = "tutorial"
            elif any(word in content for word in ["api", "documentation", "reference"]):
                doc_type = "documentation"
            else:
                doc_type = "technical_doc"

            # Determine section
            if any(
                word in content for word in ["introduction", "overview", "background"]
            ):
                section = "introduction"
            elif any(word in content for word in ["method", "approach", "algorithm"]):
                section = "methodology"
            elif any(
                word in content for word in ["result", "evaluation", "performance"]
            ):
                section = "results"
            elif any(
                word in content for word in ["conclusion", "summary", "future work"]
            ):
                section = "conclusion"
            else:
                section = "content"

            # Determine topic
            if any(
                word in content
                for word in ["machine learning", "ml", "neural", "model"]
            ):
                topic = "machine_learning"
            elif any(
                word in content for word in ["document", "parsing", "text extraction"]
            ):
                topic = "document_processing"
            elif any(
                word in content
                for word in ["nlp", "natural language", "text processing"]
            ):
                topic = "nlp"
            else:
                topic = "data_science"

            # Add enriched metadata
            enriched_doc = doc.copy()
            enriched_doc["metadata"] = {
                **doc.get("metadata", {}),
                "document_type": doc_type,
                "section": section,
                "topic": topic,
                "difficulty": "intermediate",  # Default difficulty
                "page_number": i + 1,  # Simulated page number
            }

            enriched_docs.append(enriched_doc)

        return enriched_docs

    def query_with_self_query(
        self, question: str, k: int = 10, show_extraction_details: bool = False
    ) -> str:
        """
        Perform self-query retrieval with metadata filtering.

        Args:
            question: Natural language query
            k: Number of documents to retrieve initially
            show_extraction_details: Whether to show extraction process

        Returns:
            Generated response using self-query results
        """
        # Step 1: Extract query components
        print("🧠 Extracting query components...")
        query_components = self.extract_query_components(question)

        semantic_query = query_components["semantic_query"]
        filters = query_components["filters"]

        if show_extraction_details:
            print(f"\nOriginal query: '{question}'")
            print(f"Semantic query: '{semantic_query}'")
            print(f"Extracted filters: {filters}")
            print()

        # Step 2: Retrieve documents using semantic query
        print("📚 Retrieving documents with semantic query...")
        retrieved_docs = self.rag_system.retrieve_context(semantic_query, k=k)

        # Step 3: Enrich documents with metadata (for demo purposes)
        enriched_docs = self.enrich_documents_with_metadata(retrieved_docs)

        # Step 4: Apply metadata filters
        if filters:
            print(f"🔍 Applying metadata filters: {filters}")
            filtered_docs = self.apply_metadata_filter(enriched_docs, filters)
            print(
                f"Filtered from {len(enriched_docs)} to {len(filtered_docs)} documents"
            )
        else:
            print("No metadata filters to apply")
            filtered_docs = enriched_docs

        if show_extraction_details:
            print(f"\nFiltered documents:")
            for i, doc in enumerate(filtered_docs[:5]):  # Show top 5
                metadata = doc.get("metadata", {})
                print(
                    f"  Doc {i + 1}: {metadata.get('document_type', 'unknown')} | "
                    f"{metadata.get('section', 'unknown')} | "
                    f"{metadata.get('topic', 'unknown')} | "
                    f"Sim: {doc['similarity']:.3f}"
                )
            print()

        # Step 5: Generate response using filtered documents
        if filtered_docs:
            print("🤖 Generating response with filtered documents...")
            response = self.rag_system.generate_response(question, filtered_docs[:5])
        else:
            print("⚠️ No documents match the filters. Using original results...")
            response = self.rag_system.generate_response(question, enriched_docs[:5])

        return response

    def compare_with_without_self_query(self, question: str) -> Dict[str, Any]:
        """
        Compare results with and without self-query filtering.
        """
        print("🔬 Comparing Self-Query vs Standard Retrieval")
        print("=" * 60)

        # Standard retrieval
        print("\n1️⃣ Standard Retrieval:")
        print("-" * 40)
        standard_context = self.rag_system.retrieve_context(question, k=5)
        standard_response = self.rag_system.generate_response(
            question, standard_context
        )

        print(f"Retrieved {len(standard_context)} documents")
        for i, doc in enumerate(standard_context):
            print(f"  Doc {i + 1}: Similarity {doc['similarity']:.3f}")

        # Self-query retrieval
        print("\n2️⃣ Self-Query Retrieval:")
        print("-" * 40)
        self_query_response = self.query_with_self_query(
            question, k=10, show_extraction_details=True
        )

        return {
            "question": question,
            "standard_response": standard_response,
            "self_query_response": self_query_response,
            "standard_context": standard_context,
        }


def demonstrate_self_query():
    """
    Demonstrate the self-query technique with practical examples.
    """
    print("🧠 Self-Query Demonstration")
    print("=" * 50)

    # Initialize self-query system
    self_query = SelfQuery()

    # Make sure we have documents in the vector store
    doc_count = self_query.rag_system.vector_store.get_document_count()
    if doc_count == 0:
        print("No documents found in vector store. Ingesting sample document...")
        from rag.config import DOCLING_PAPER_URL  # type: ignore

        self_query.rag_system.ingest_document(DOCLING_PAPER_URL)

    print(
        f"Vector store contains {self_query.rag_system.vector_store.get_document_count()} documents"
    )

    # Test queries that benefit from self-query filtering
    test_queries = [
        "Find research papers about machine learning methodology",
        "Show me tutorial content about document processing for beginners",
        "What are the results from advanced NLP experiments?",
        "Give me introduction sections about data science",
    ]

    print(f"\n🧪 Testing Self-Query with Sample Queries")
    print("=" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"TEST {i}: {query}")
        print("=" * 60)

        # Show comparison between standard and self-query retrieval
        comparison = self_query.compare_with_without_self_query(query)

        print(f"\n📊 RESULTS COMPARISON:")
        print("-" * 30)
        print(f"Standard Response: {comparison['standard_response'][:200]}...")
        print(f"\nSelf-Query Response: {comparison['self_query_response'][:200]}...")

        print(f"\n" + "=" * 60)

    # Demonstrate query component extraction
    print(f"\n🎯 Query Component Extraction Examples")
    print("=" * 40)

    example_queries = [
        "Find research papers about machine learning",
        "Show me beginner tutorials on document processing",
        "What are the results from page 5?",
        "Give me advanced methodology sections",
    ]

    for query in example_queries:
        print(f"\nQuery: '{query}'")
        components = self_query.extract_query_components(query)
        print(f"  Semantic: '{components['semantic_query']}'")
        print(f"  Filters: {components['filters']}")


if __name__ == "__main__":
    demonstrate_self_query()
