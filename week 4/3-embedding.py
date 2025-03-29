from typing import List

import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper

# --------------------------------------------------------------
# Exercise 3: Building a Vector Database with LanceDB
# --------------------------------------------------------------
"""
In this exercise, we'll create a vector database using LanceDB to store and retrieve embeddings.

As discussed in Week 4, vector databases are essential for efficient storage and retrieval 
of embeddings. While there are many options available (Pinecone, Weaviate, Qdrant, etc.), 
we're using LanceDB for several key reasons:

1. Embedded database - lives entirely within your project with no external infrastructure
2. Simple setup - minimal configuration required to get started
3. Powerful API - supports sophisticated search strategies out of the box
4. Open source - can be used in any project without licensing concerns

This exercise will walk through the complete RAG pipeline:
1. Document extraction
2. Chunking
3. Embedding generation
4. Vector storage
"""

load_dotenv()

# Initialize OpenAI client (make sure you have OPENAI_API_KEY in your environment variables)
client = OpenAI()

tokenizer = OpenAITokenizerWrapper()  # Load our custom tokenizer for OpenAI
MAX_TOKENS = 8191  # text-embedding-3-large's maximum context length

# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------
"""
First, we'll extract a research paper from arXiv using Docling.
The standardized document model will make chunking more effective.
"""

converter = DocumentConverter()
result = converter.convert("https://bitcoin.org/bitcoin.pdf")

# --------------------------------------------------------------
# Apply hybrid chunking
# --------------------------------------------------------------
"""
Next, we'll apply the HybridChunker to create semantically meaningful chunks.
This chunker balances document structure with token limits to create optimal chunks.
"""

chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)

chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)

# --------------------------------------------------------------
# Create a LanceDB database and table
# --------------------------------------------------------------
"""
Now we'll set up our vector database using LanceDB.
The embedded database will be stored in the data/lancedb directory.

We're using OpenAI's text-embedding-3-large model to generate embeddings.
LanceDB handles the embedding generation automatically when we add documents.
"""

# Create a LanceDB database
db = lancedb.connect("data/lancedb")

# Get the OpenAI embedding function
func = get_registry().get("openai").create(name="text-embedding-3-large")


# Define a simplified metadata schema
class ChunkMetadata(LanceModel):
    """
    You must order the fields in alphabetical order.
    This is a requirement of the Pydantic implementation.
    """

    filename: str | None
    page_numbers: List[int] | None
    title: str | None


# Define the main Schema
class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadata


table = db.create_table("crypto", schema=Chunks, mode="overwrite")

# --------------------------------------------------------------
# Prepare the chunks for the table
# --------------------------------------------------------------
"""
Before adding chunks to the database, we need to format them properly.
We'll extract relevant metadata like filename, page numbers, and section titles
to enhance our retrieval capabilities later.
"""

# Create table with processed chunks
processed_chunks = [
    {
        "text": chunk.text,
        "metadata": {
            "filename": chunk.meta.origin.filename,
            "page_numbers": [
                page_no
                for page_no in sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    )
                )
            ]
            or None,
            "title": chunk.meta.headings[0] if chunk.meta.headings else None,
        },
    }
    for chunk in chunks
]

# --------------------------------------------------------------
# Add the chunks to the table (automatically embeds the text)
# --------------------------------------------------------------
"""
When we add chunks to the table, LanceDB will automatically:
1. Send the text to OpenAI to generate embeddings
2. Store both the original text and the vector embeddings
3. Index the vectors for efficient similarity search
"""

table.add(processed_chunks)

# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------
"""
Let's verify our data was properly stored by loading the table
and checking the number of rows.
"""

table.to_pandas()
table.count_rows()

# --------------------------------------------------------------
# Your Turn: Insert the Ethereum whitepaper
# --------------------------------------------------------------
"""
1. Use this code to insert the Ethereum whitepaper
2. URL: https://ethereum.org/content/whitepaper/whitepaper-pdf/Ethereum_Whitepaper_-_Buterin_2014.pdf
3. Validate the data was inserted correctly by running
"""
