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
    Implements Cohere Reranking technique for RAG systems using Wikipedia articles.
    """

    def __init__(self):
        """Initialize the Cohere reranking system."""
        self.rag_system = RAGSystem()
        self.cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
        self.default_model = "rerank-english-v3.0"

    def get_data(self):
        """Clear database and insert Wikipedia article chunks."""
        print("Clearing existing documents...")
        self.rag_system.vector_store.clear_all_documents()

        print("Loading Wikipedia articles...")

        urls = [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Data_science",
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://en.wikipedia.org/wiki/Natural_language_processing",
            "https://en.wikipedia.org/wiki/Computer_vision",
            "https://en.wikipedia.org/wiki/Deep_learning",
            "https://en.wikipedia.org/wiki/Big_data",
            "https://en.wikipedia.org/wiki/Cloud_computing",
            "https://en.wikipedia.org/wiki/Internet_of_things",
            "https://en.wikipedia.org/wiki/Robotics",
        ]

        total_chunks = 0
        for i, url in enumerate(urls, 1):
            print(f"Processing {i}/{len(urls)}: {url.split('/')[-1].replace('_', ' ')}")
            chunks = self.rag_system.document_processor.process_document(url)

            for chunk in chunks:
                embedding = self.rag_system.embedding_service.create_embedding(
                    chunk["content"]
                )
                self.rag_system.vector_store.add_document(
                    content=chunk["content"],
                    embedding=embedding,
                    metadata=chunk["metadata"],
                )

            total_chunks += len(chunks)
            print(f"  Added {len(chunks)} chunks")

        print(
            f"✅ Inserted {total_chunks} total chunks from {len(urls)} Wikipedia articles"
        )

    def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = None,
    ) -> List[Dict[str, Any]]:
        """Rerank documents using Cohere's rerank API."""
        if not documents:
            return []

        # Prepare documents for Cohere API
        doc_texts = [doc["content"] for doc in documents]

        try:
            rerank_response = self.cohere_client.rerank(
                model=self.default_model,
                query=query,
                documents=doc_texts,
                top_n=top_n,
                return_documents=True,
            )

            # Process the reranked results
            reranked_docs = []
            for result in rerank_response.results:
                original_doc = documents[result.index].copy()
                original_doc["rerank_score"] = result.relevance_score
                original_doc["original_rank"] = result.index + 1
                reranked_docs.append(original_doc)

            return reranked_docs

        except Exception as e:
            print(f"Error during reranking: {e}")
            return documents

    def compare_with_without_reranking(
        self, question: str, initial_k: int = 15, final_k: int = 5
    ):
        """Compare results with and without reranking."""
        print(f"\n{'=' * 60}")
        print(f"COMPARISON: {question}")
        print("=" * 60)

        # Standard retrieval
        print("\n1️⃣ STANDARD RETRIEVAL:")
        print("-" * 30)
        standard_docs = self.rag_system.retrieve_context(question, k=final_k)
        print(f"Retrieved top {len(standard_docs)} documents by similarity")

        # Retrieval with reranking
        print("\n2️⃣ RETRIEVAL WITH RERANKING:")
        print("-" * 30)

        # Get more documents initially
        initial_docs = self.rag_system.retrieve_context(question, k=initial_k)
        print(f"Initial retrieval: {len(initial_docs)} documents")

        # Rerank them
        reranked_docs = self.rerank_documents(question, initial_docs, top_n=final_k)
        print(f"Reranked to top {len(reranked_docs)} most relevant documents")

        # Show ranking changes - this is the key insight!
        print("\n📊 RANKING CHANGES:")
        print("-" * 30)
        significant_changes = []
        new_discoveries = []

        for i, doc in enumerate(reranked_docs, 1):
            original_rank = doc.get("original_rank", 0)
            rerank_score = doc.get("rerank_score", 0)
            similarity = doc.get("similarity", 0)
            change = original_rank - i

            if original_rank > 5:
                direction = f"🆕 NEW (was #{original_rank})"
                new_discoveries.append(
                    f"Rank {i} is a new discovery from position #{original_rank}!"
                )
            elif change > 0:
                direction = f"📈 UP {change}"
                if change >= 3:
                    significant_changes.append(
                        f"Rank {i} jumped up {change} positions!"
                    )
            elif change < 0:
                direction = f"📉 DOWN {abs(change)}"
            else:
                direction = "➡️ SAME"

            print(
                f"  Rank {i}: {direction:15} | Rerank: {rerank_score:.3f} | Sim: {similarity:.3f} | Was #{original_rank}"
            )

        if new_discoveries:
            print("\n🎉 NEW DISCOVERIES:")
            for discovery in new_discoveries:
                print(f"  • {discovery}")

        if significant_changes:
            print("\n🎯 SIGNIFICANT IMPROVEMENTS:")
            for change in significant_changes:
                print(f"  • {change}")

        if not new_discoveries and not significant_changes:
            print("\n📊 Minor reordering - reranking refined the results")

        return {
            "standard_docs": standard_docs,
            "reranked_docs": reranked_docs,
            "initial_docs": initial_docs,
        }


def simple_reranking_example():
    reranker = CohereReranking()
    reranked_docs = reranker.rerank_documents(
        query="What is artificial intelligence?",
        documents=[
            {
                "content": "AI is a branch of computer science that focuses on creating machines that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language understanding."
            },
            {
                "content": "AI systems can learn from data, adapt to new situations, and improve their performance over time."
            },
            {
                "content": "AI has applications in various fields, including healthcare, finance, education, and entertainment."
            },
            {
                "content": "AI is a rapidly growing field with many potential applications and opportunities for innovation."
            },
            {
                "content": "Artificial intelligence (AI) is a broad field that encompasses various techniques and applications."
            },
        ],
        top_n=5,
    )
    print("Reranked documents:")
    for doc in reranked_docs:
        print(f"Score: {doc['rerank_score']}")
        print(f"Content: {doc['content']}")
        print()


def demonstrate_cohere_reranking():
    """Demonstrate Cohere reranking with Wikipedia articles."""
    print("🔄 Cohere Reranking Demonstration")
    print("=" * 50)

    try:
        # Initialize reranking system
        reranker = CohereReranking()

        # Setup Wikipedia data
        reranker.get_data()  # This takes ~3 minutes to process

        # Demonstrate reranking with a simple example
        simple_reranking_example()

        # Test queries that should show clear reranking benefits
        test_queries = [
            "How does artificial intelligence help with image recognition?",
            "What are the business applications of big data analytics?",
            "How do robots use computer vision for navigation?",
        ]

        print(f"\n🧪 Testing {len(test_queries)} queries that benefit from reranking")

        for query in test_queries:
            reranker.compare_with_without_reranking(query, initial_k=25, final_k=5)

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your Cohere API key and internet connection.")


if __name__ == "__main__":
    demonstrate_cohere_reranking()
