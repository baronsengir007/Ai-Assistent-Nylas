"""
Document Chunking for Vector Databases

This file demonstrates how to chunk documents for vector database ingestion
using Docling's HybridChunker with tokenization-aware refinements.
"""

import tiktoken
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer


def setup_tokenizer(model_name="text-embedding-3-large", max_tokens=8191):
    """Set up the OpenAI tokenizer for chunking."""
    tiktoken_encoder = tiktoken.encoding_for_model(model_name)
    return OpenAITokenizer(tokenizer=tiktoken_encoder, max_tokens=max_tokens)


def create_chunker(tokenizer):
    """Create a HybridChunker with the specified tokenizer."""
    return HybridChunker(tokenizer=tokenizer)


def chunk_document(doc, chunker):
    """Chunk a document using the specified chunker."""
    return list(chunker.chunk(dl_doc=doc))


def analyze_chunk(chunk, chunker, tokenizer):
    """Analyze a single chunk, including token counts and contextualization."""
    raw_text = chunk.text
    contextualized_text = chunker.contextualize(chunk=chunk)

    raw_tokens = tokenizer.tokenizer.encode(raw_text)
    contextualized_tokens = tokenizer.tokenizer.encode(contextualized_text)

    return {
        "raw_text": raw_text,
        "contextualized_text": contextualized_text,
        "raw_tokens": len(raw_tokens),
        "contextualized_tokens": len(contextualized_tokens),
        "token_overhead": len(contextualized_tokens) - len(raw_tokens),
    }


def analyze_chunks(chunks, chunker, tokenizer, max_samples=5):
    """Analyze multiple chunks, including token counts and contextualization."""
    results = []
    total_raw_tokens = 0
    total_contextualized_tokens = 0

    for i, chunk in enumerate(chunks):
        analysis = analyze_chunk(chunk, chunker, tokenizer)
        results.append(analysis)

        total_raw_tokens += analysis["raw_tokens"]
        total_contextualized_tokens += analysis["contextualized_tokens"]

        if i < max_samples:
            print(f"Chunk {i}:")
            print(f"  Raw tokens: {analysis['raw_tokens']}")
            print(f"  Contextualized tokens: {analysis['contextualized_tokens']}")
            print(f"  Token overhead: +{analysis['token_overhead']}")
            print(f"  Raw preview: {repr(analysis['raw_text'][:100])}...")
            print(
                f"  Contextualized preview: {repr(analysis['contextualized_text'][:100])}..."
            )
            print()

    if len(chunks) > max_samples:
        print(f"... and {len(chunks) - max_samples} more chunks")

    return {
        "total_chunks": len(chunks),
        "total_raw_tokens": total_raw_tokens,
        "total_contextualized_tokens": total_contextualized_tokens,
        "avg_raw_tokens": total_raw_tokens / len(chunks),
        "avg_contextualized_tokens": total_contextualized_tokens / len(chunks),
        "token_overhead_percent": (
            (total_contextualized_tokens - total_raw_tokens) / total_raw_tokens * 100
        ),
    }


def main():
    # Convert document
    doc = DocumentConverter().convert("https://arxiv.org/pdf/2408.09869").document
    print(f"Document converted: {len(doc.pages)} pages")

    # Set up tokenizer and chunker
    tokenizer = setup_tokenizer()
    chunker = create_chunker(tokenizer)

    # Chunk document
    chunks = chunk_document(doc, chunker)
    print(f"Document split into {len(chunks)} chunks")

    # Analyze chunks
    summary = analyze_chunks(chunks, chunker, tokenizer)

    # Print summary
    print("\nChunking Summary:")
    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Total raw tokens: {summary['total_raw_tokens']}")
    print(f"Total contextualized tokens: {summary['total_contextualized_tokens']}")
    print(f"Average tokens per raw chunk: {summary['avg_raw_tokens']:.1f}")
    print(
        f"Average tokens per contextualized chunk: {summary['avg_contextualized_tokens']:.1f}"
    )
    print(
        f"Token overhead: +{summary['total_contextualized_tokens'] - summary['total_raw_tokens']} ({summary['token_overhead_percent']:.1f}%)"
    )


if __name__ == "__main__":
    main()
