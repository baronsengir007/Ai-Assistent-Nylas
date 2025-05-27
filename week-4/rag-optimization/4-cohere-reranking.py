"""
Cohere Reranking - RAG Optimization Technique #4

Reranking improves RAG by reordering retrieved documents based on their relevance
to the query. Cohere's rerank model uses deep learning to evaluate query-document
alignment, providing more nuanced relevance scoring than simple similarity metrics.

Key Benefits:
- Improves precision by reordering results by relevance
- Reduces retrieval failure rate by up to 67% when combined with other techniques
- Works with any initial retrieval method (semantic, keyword, hybrid)
- Provides more nuanced relevance scoring than cosine similarity
"""

import sys
from pathlib import Path

# Add the rag-pipeline to the path so we can import from it
sys.path.append(str(Path(__file__).parent.parent) + "/rag-pipeline")

import os
from rag.rag_system import RAGSystem  # type: ignore
from typing import List, Dict, Any
import cohere


class CohereReranking:
    """
    Implements Cohere Reranking technique for RAG systems.

    This class uses Cohere's rerank API to reorder retrieved documents
    based on their relevance to the query, improving result quality.
    """

    def __init__(self, cohere_api_key: str = None):
        """
        Initialize the Cohere reranking system.

        Args:
            cohere_api_key: Cohere API key. If None, will try to get from environment.
        """
        self.rag_system = RAGSystem()

        # Initialize Cohere client
        if cohere_api_key is None:
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if cohere_api_key is None:
                raise ValueError(
                    "Cohere API key is required. Set COHERE_API_KEY environment variable "
                    "or pass it as a parameter. Get your free API key at: "
                    "https://dashboard.cohere.com/api-keys"
                )

        self.cohere_client = cohere.Client(cohere_api_key)

        # Available Cohere rerank models
        self.available_models = {
            "rerank-english-v3.0": "Latest English rerank model with 4k context length",
            "rerank-multilingual-v3.0": "Multilingual rerank model supporting 100+ languages",
            "rerank-english-v2.0": "Previous generation English model",
            "rerank-multilingual-v2.0": "Previous generation multilingual model",
        }

        # Default model
        self.default_model = "rerank-english-v3.0"

    def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        model: str = None,
        top_n: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using Cohere's rerank API.

        Args:
            query: The search query
            documents: List of documents to rerank
            model: Cohere rerank model to use
            top_n: Number of top documents to return (None = return all)

        Returns:
            Reranked list of documents with relevance scores
        """
        if not documents:
            return []

        if model is None:
            model = self.default_model

        # Prepare documents for Cohere API
        # Cohere expects a list of strings (document texts)
        doc_texts = []
        for doc in documents:
            # Use the content field for reranking
            doc_texts.append(doc["content"])

        try:
            # Call Cohere rerank API
            print(f"🔄 Reranking {len(documents)} documents with Cohere {model}...")

            rerank_response = self.cohere_client.rerank(
                model=model,
                query=query,
                documents=doc_texts,
                top_n=top_n,
                return_documents=True,
            )

            # Process the reranked results
            reranked_docs = []
            for result in rerank_response.results:
                # Get the original document
                original_doc = documents[result.index].copy()

                # Add rerank information
                original_doc["rerank_score"] = result.relevance_score
                original_doc["original_index"] = result.index
                original_doc["reranked_text"] = (
                    result.document.text
                    if hasattr(result.document, "text")
                    else doc_texts[result.index]
                )

                # Update metadata
                if "metadata" not in original_doc:
                    original_doc["metadata"] = {}
                original_doc["metadata"]["rerank_score"] = result.relevance_score
                original_doc["metadata"]["rerank_model"] = model
                original_doc["metadata"]["original_rank"] = result.index + 1

                reranked_docs.append(original_doc)

            print(f"✅ Reranking complete. Returned {len(reranked_docs)} documents")
            return reranked_docs

        except Exception as e:
            print(f"❌ Error during reranking: {e}")
            print("Falling back to original document order...")
            return documents

    def retrieve_and_rerank(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 5,
        model: str = None,
        show_rerank_details: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents and then rerank them.

        Args:
            query: Search query
            initial_k: Number of documents to retrieve initially
            final_k: Number of documents to return after reranking
            model: Cohere rerank model to use
            show_rerank_details: Whether to show reranking process details

        Returns:
            Reranked list of top documents
        """
        # Step 1: Initial retrieval
        print(f"📚 Initial retrieval: getting top {initial_k} documents...")
        initial_docs = self.rag_system.retrieve_context(query, k=initial_k)

        if show_rerank_details:
            print(f"\nInitial retrieval results (top 5):")
            for i, doc in enumerate(initial_docs[:5]):
                print(
                    f"  {i + 1}. Similarity: {doc['similarity']:.3f} | Content: {doc['content'][:100]}..."
                )

        # Step 2: Rerank documents
        reranked_docs = self.rerank_documents(
            query=query, documents=initial_docs, model=model, top_n=final_k
        )

        if show_rerank_details:
            print(f"\nReranked results (top {len(reranked_docs)}):")
            for i, doc in enumerate(reranked_docs):
                original_rank = doc.get("original_index", -1) + 1
                rerank_score = doc.get("rerank_score", 0)
                similarity = doc.get("similarity", 0)
                print(
                    f"  {i + 1}. Rerank: {rerank_score:.3f} | Original rank: {original_rank} | "
                    f"Similarity: {similarity:.3f} | Content: {doc['content'][:100]}..."
                )

        return reranked_docs

    def query_with_reranking(
        self,
        question: str,
        initial_k: int = 20,
        final_k: int = 5,
        model: str = None,
        show_rerank_details: bool = False,
    ) -> str:
        """
        Perform a complete RAG query with reranking.

        Args:
            question: User's question
            initial_k: Number of documents to retrieve initially
            final_k: Number of documents to use for generation
            model: Cohere rerank model to use
            show_rerank_details: Whether to show reranking details

        Returns:
            Generated response using reranked documents
        """
        # Retrieve and rerank documents
        reranked_docs = self.retrieve_and_rerank(
            query=question,
            initial_k=initial_k,
            final_k=final_k,
            model=model,
            show_rerank_details=show_rerank_details,
        )

        # Generate response using reranked documents
        if reranked_docs:
            print("🤖 Generating response with reranked documents...")
            response = self.rag_system.generate_response(question, reranked_docs)
        else:
            print("⚠️ No documents available for generation")
            response = "I couldn't find relevant information to answer your question."

        return response

    def compare_with_without_reranking(
        self, question: str, initial_k: int = 20, final_k: int = 5
    ) -> Dict[str, Any]:
        """
        Compare results with and without reranking.

        Returns:
            Dictionary containing both results for comparison
        """
        print("🔬 Comparing Reranking vs Standard Retrieval")
        print("=" * 60)

        # Standard retrieval (without reranking)
        print(f"\n1️⃣ Standard Retrieval (Top {final_k}):")
        print("-" * 40)
        standard_docs = self.rag_system.retrieve_context(question, k=final_k)
        standard_response = self.rag_system.generate_response(question, standard_docs)

        print(f"Retrieved {len(standard_docs)} documents")
        for i, doc in enumerate(standard_docs):
            print(f"  Doc {i + 1}: Similarity {doc['similarity']:.3f}")

        # Reranked retrieval
        print(f"\n2️⃣ Reranked Retrieval (Top {initial_k} → Top {final_k}):")
        print("-" * 40)
        reranked_response = self.query_with_reranking(
            question=question,
            initial_k=initial_k,
            final_k=final_k,
            show_rerank_details=True,
        )

        return {
            "question": question,
            "standard_response": standard_response,
            "reranked_response": reranked_response,
            "standard_docs": standard_docs,
        }

    def demonstrate_rerank_models(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> None:
        """
        Demonstrate different Cohere rerank models.
        """
        print(f"\n🎯 Comparing Different Rerank Models")
        print("=" * 50)
        print(f"Query: '{query}'")
        print(f"Documents to rerank: {len(documents)}")

        for model_name, description in self.available_models.items():
            print(f"\n📊 Model: {model_name}")
            print(f"Description: {description}")
            print("-" * 40)

            try:
                reranked = self.rerank_documents(
                    query=query, documents=documents, model=model_name, top_n=3
                )

                for i, doc in enumerate(reranked):
                    print(
                        f"  {i + 1}. Score: {doc['rerank_score']:.3f} | "
                        f"Content: {doc['content'][:80]}..."
                    )

            except Exception as e:
                print(f"  ❌ Error with {model_name}: {e}")


def demonstrate_cohere_reranking():
    """
    Demonstrate the Cohere reranking technique with practical examples.
    """
    print("🔄 Cohere Reranking Demonstration")
    print("=" * 50)

    # Check for Cohere API key
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        print("❌ COHERE_API_KEY environment variable not set!")
        print("Please set your Cohere API key:")
        print("export COHERE_API_KEY='your-api-key-here'")
        print("\nGet your free API key at: https://dashboard.cohere.com/api-keys")
        return

    try:
        # Initialize reranking system
        reranker = CohereReranking(cohere_api_key)

        # Make sure we have documents in the vector store
        doc_count = reranker.rag_system.vector_store.get_document_count()
        if doc_count == 0:
            print("No documents found in vector store. Ingesting sample document...")
            from rag.config import DOCLING_PAPER_URL

            reranker.rag_system.ingest_document(DOCLING_PAPER_URL)

        print(
            f"Vector store contains {reranker.rag_system.vector_store.get_document_count()} documents"
        )

        # Test queries that benefit from reranking
        test_queries = [
            "What is document parsing and how does it work?",
            "How does Docling handle different document formats?",
            "What are the performance benchmarks and evaluation metrics?",
        ]

        print(f"\n🧪 Testing Cohere Reranking with Sample Queries")
        print("=" * 50)

        for i, query in enumerate(test_queries, 1):
            print(f"\n{'=' * 60}")
            print(f"TEST {i}: {query}")
            print("=" * 60)

            # Show comparison between standard and reranked retrieval
            comparison = reranker.compare_with_without_reranking(
                question=query, initial_k=15, final_k=5
            )

            print(f"\n📊 RESULTS COMPARISON:")
            print("-" * 30)
            print(f"Standard Response: {comparison['standard_response'][:200]}...")
            print(f"\nReranked Response: {comparison['reranked_response'][:200]}...")

            print(f"\n" + "=" * 60)

        # Demonstrate different rerank models
        print(f"\n🎯 Available Cohere Rerank Models")
        print("=" * 40)
        for model, desc in reranker.available_models.items():
            print(f"• {model}: {desc}")

        # Test with a sample query and documents
        sample_query = "document processing and parsing"
        sample_docs = reranker.rag_system.retrieve_context(sample_query, k=5)

        if sample_docs:
            reranker.demonstrate_rerank_models(sample_query, sample_docs)

    except Exception as e:
        print(f"❌ Error initializing Cohere reranking: {e}")
        print("Please check your API key and internet connection.")


if __name__ == "__main__":
    demonstrate_cohere_reranking()
