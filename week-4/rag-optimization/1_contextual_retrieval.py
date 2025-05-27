"""
Contextual Retrieval - RAG Optimization Technique #1

This technique, introduced by Anthropic, enhances RAG by adding contextual information
to each chunk before embedding. This helps preserve important context that might be
lost when documents are split into smaller pieces.

Link: https://www.anthropic.com/news/contextual-retrieval

Key Benefits:
- Reduces retrieval failure rate by 35% (Anthropic's findings)
- Preserves document context that gets lost during chunking
- Improves semantic understanding of isolated chunks
"""

import sys
from pathlib import Path

# Add the rag-pipeline to the path so we can import from it
sys.path.append(str(Path(__file__).parent.parent) + "/rag-pipeline")

from openai import OpenAI
from rag.rag_system import RAGSystem  # type: ignore
from rag.config import OPENAI_API_KEY  # type: ignore
from typing import List, Dict, Any


class ContextualRetrieval:
    """
    Implements Anthropic's Contextual Retrieval technique.

    This class adds contextual information to document chunks before embedding,
    helping to preserve important context that might be lost during chunking.
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.rag_system = RAGSystem()

    def generate_context_for_chunk(
        self, document_content: str, chunk_content: str
    ) -> str:
        """
        Generate contextual information for a chunk using the full document.

        This is the core of Anthropic's contextual retrieval technique.
        """
        prompt = f"""
        <document>
        {document_content}
        </document>
        
        Here is the chunk we want to situate within the whole document:
        <chunk>
        {chunk_content}
        </chunk>
        
        Please give a short succinct context to situate this chunk within the overall 
        document for the purposes of improving search retrieval of the chunk. 
        Answer only with the succinct context and nothing else.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating context: {e}")
            return ""

    def process_document_with_context(self, source: str) -> List[Dict[str, Any]]:
        """
        Process a document by adding contextual information to each chunk.

        Args:
            source: Document source (URL or file path)

        Returns:
            List of chunks with added contextual information
        """
        print("Processing document with contextual retrieval...")

        # First, get the original document content
        chunks = self.rag_system.document_processor.process_document(source)

        # Get the full document text for context generation
        # Note: In a real implementation, you'd want to store the full document
        # For this demo, we'll reconstruct it from chunks (not ideal but functional)
        full_document = "\n\n".join([chunk["content"] for chunk in chunks])

        # Add contextual information to each chunk
        contextual_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Adding context to chunk {i + 1}/{len(chunks)}...")

            # Generate context for this chunk
            context = self.generate_context_for_chunk(full_document, chunk["content"])

            # Prepend context to the original chunk content
            if context:
                contextual_content = f"{context}\n\n{chunk['content']}"
            else:
                contextual_content = chunk["content"]

            # Create new chunk with contextual content
            contextual_chunk = {
                "content": contextual_content,
                "metadata": {
                    **chunk.get("metadata", {}),
                    "original_content": chunk["content"],
                    "added_context": context,
                    "chunk_index": i,
                },
            }
            contextual_chunks.append(contextual_chunk)

        return contextual_chunks

    def ingest_with_contextual_retrieval(self, source: str):
        """
        Ingest a document using contextual retrieval technique.
        """
        # Process document with contextual information
        contextual_chunks = self.process_document_with_context(source)

        # Add embeddings to the contextual chunks
        chunks_with_embeddings = (
            self.rag_system.embedding_service.add_embeddings_to_chunks(
                contextual_chunks
            )
        )

        # Store in vector database
        print("Storing contextual chunks in vector database...")
        self.rag_system.vector_store.add_documents(chunks_with_embeddings)

        # Print statistics
        print(f"\nContextual retrieval ingestion complete!")
        print(f"Total chunks processed: {len(contextual_chunks)}")
        print(
            f"Total documents in store: {self.rag_system.vector_store.get_document_count()}"
        )

        return contextual_chunks

    def query_with_context_info(
        self, question: str, show_context_details: bool = False
    ) -> str:
        """
        Query the RAG system and optionally show how contextual retrieval helped.
        """
        # Retrieve context using the existing RAG system
        context = self.rag_system.retrieve_context(question)

        if show_context_details:
            print("\n" + "=" * 50)
            print("CONTEXTUAL RETRIEVAL DETAILS")
            print("=" * 50)
            for i, doc in enumerate(context, 1):
                print(f"\nChunk {i}:")
                print(f"Similarity: {doc['similarity']:.3f}")
                if "added_context" in doc.get("metadata", {}):
                    print(f"Added Context: {doc['metadata']['added_context']}")
                    print(
                        f"Original Content: {doc['metadata']['original_content'][:200]}..."
                    )
                else:
                    print(f"Content: {doc['content'][:200]}...")
                print("-" * 30)

        # Generate response
        response = self.rag_system.generate_response(question, context)
        return response


def demonstrate_contextual_retrieval():
    """
    Demonstrate the contextual retrieval technique with a practical example.
    """
    print("🔍 Contextual Retrieval Demonstration")
    print("=" * 50)

    # Initialize contextual retrieval system
    contextual_rag = ContextualRetrieval()

    # Clear any existing documents
    contextual_rag.rag_system.vector_store.clear_all_documents()

    # Use the same document source as the main RAG pipeline
    from rag.config import DOCLING_PAPER_URL  # type: ignore

    print(f"Ingesting document with contextual retrieval: {DOCLING_PAPER_URL}")

    # Ingest document using contextual retrieval
    contextual_chunks = contextual_rag.ingest_with_contextual_retrieval(
        DOCLING_PAPER_URL
    )

    # Show example of how context was added
    print("\n📄 Example of Contextual Enhancement:")
    print("-" * 40)
    if contextual_chunks:
        example_chunk = contextual_chunks[2]  # Show the 3rd chunk as example
        print("Original chunk content:")
        print(f"'{example_chunk['metadata']['original_content'][:150]}...'")
        print(f"\nAdded context:")
        print(f"'{example_chunk['metadata']['added_context']}'")
        print(f"\nFinal contextual chunk:")
        print(f"'{example_chunk['content'][:200]}...'")

    # Test queries
    test_queries = [
        "What is document parsing and why is it important?",
        "How does Docling handle different document formats?",
        "What are the main components of the Docling architecture?",
    ]

    print(f"\n🤖 Testing Contextual Retrieval with Sample Queries")
    print("=" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 30)

        response = contextual_rag.query_with_context_info(
            query, show_context_details=True
        )
        print(f"\nResponse: {response}")
        print("\n" + "=" * 50)


if __name__ == "__main__":
    demonstrate_contextual_retrieval()
