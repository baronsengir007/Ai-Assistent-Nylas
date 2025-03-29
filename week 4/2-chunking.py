from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper

# --------------------------------------------------------------
# Exercise 2: Document Chunking Strategies
# --------------------------------------------------------------
"""
In this exercise, we'll implement different chunking strategies using Docling.

As covered in Week 4, chunking is a critical step in the RAG pipeline that balances several factors:
- Chunk Size: Larger chunks contain more context but may dilute relevance
- Chunk Overlap: Some overlap ensures concepts that cross boundaries aren't lost
- Semantic Coherence: Ideally, chunks should represent complete thoughts or sections

Docling provides advanced chunking capabilities through its HybridChunker, which combines:
1. Structure-aware chunking (preserving document hierarchy)
2. Token-aware sizing (ensuring chunks fit embedding model limits)
3. Intelligent merging (combining related content when appropriate)

In this exercise, we'll chunk the Bitcoin whitepaper and explore the results.
"""

load_dotenv()

# Initialize OpenAI client (make sure you have OPENAI_API_KEY in your environment variables)
client = OpenAI()

# Initialize a tokenizer that matches our embedding model
tokenizer = OpenAITokenizerWrapper()  # Load our custom tokenizer for OpenAI
MAX_TOKENS = 8191  # text-embedding-3-large's maximum context length

# --------------------------------------------------------------
# Extract the document (same as before)
# --------------------------------------------------------------
converter = DocumentConverter()
result = converter.convert("https://bitcoin.org/bitcoin.pdf")
document = result.document

print(f"Document title: {document.name}")

# --------------------------------------------------------------
# Apply hybrid chunking
# --------------------------------------------------------------
"""
The HybridChunker uses a sophisticated approach that:
1. Respects document structure (sections, paragraphs, lists)
2. Ensures chunks don't exceed token limits
3. Preserves context by attaching headers and captions
4. Merges small, related chunks when possible
"""

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)

chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)

print(f"\nTotal chunks created: {len(chunks)}")

# --------------------------------------------------------------
# Experiment with different chunking strategies
# --------------------------------------------------------------
"""
Now let's compare different chunking approaches to see how they affect 
the resulting chunks.
"""

# 1. Change the maximum token size
small_chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=500,  # Much smaller chunks
    merge_peers=True,
)
small_chunks = list(small_chunker.chunk(dl_doc=result.document))
print(f"\nWith max_tokens=1000: {len(small_chunks)} chunks created")

# 2. Disable peer merging
no_merge_chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=False,  # Don't merge related chunks
)
no_merge_chunks = list(no_merge_chunker.chunk(dl_doc=result.document))
print(f"With merge_peers=False: {len(no_merge_chunks)} chunks created")
