from docling.document_converter import DocumentConverter

# --------------------------------------------------------------
# Exercise 1: Document Extraction with Docling
# --------------------------------------------------------------
"""
In this exercise, we'll explore document parsing and standardized data models,
the first critical step in our RAG pipeline.

As discussed in Week 4, using standardized document models provides several key advantages:
1. Format independence - process multiple document types consistently
2. Consistent metadata - extract and represent metadata uniformly 
3. Simplified pipeline logic - downstream processing needs to handle only one document representation
4. Future-proofing - new document formats can be integrated by adding new parsers

Docling provides this standardization out-of-the-box, converting various document formats
into a consistent data structure with normalized sections, metadata, and content.

Full documentation: https://docling-project.github.io/docling/

In this exercise, we'll parse the Bitcoin whitepaper and explore the resulting document model.
"""

# Initialize the document converter
converter = DocumentConverter()

# --------------------------------------------------------------
# Basic PDF extraction using docling
# --------------------------------------------------------------
# Extract the Bitcoin whitepaper - a common test document for RAG systems
# Notice how we can pass a URL and Docling handles the downloading and extraction
result = converter.convert("https://bitcoin.org/bitcoin.pdf")

# Get the standardized document object
document = result.document

# Export to different formats for different use cases
markdown_output = document.export_to_markdown()
json_output = document.export_to_dict()

# View the document in markdown format
print("Document in Markdown format:")
print(markdown_output[:500] + "...\n")  # Print just the first 500 chars

# --------------------------------------------------------------
# Explore the standardized data model
# --------------------------------------------------------------
"""
Understanding the document model is crucial for effective RAG implementation.
The standardized structure lets us access document components (title, sections, etc.)
programmatically, which will be essential for chunking in the next exercise.
"""

# Examine the document type
print(f"Document type: {type(document)}\n")

print(document.model_dump_json(indent=2))

# Print document metadata
print(f"Document title: {document.name}")
print(f"Document type: {document.origin.mimetype}")
print(f"Document file_name: {document.origin.filename}")


# Dump the full document structure to JSON for detailed examination
print("\nFull document model (truncated):")
print(document.model_dump_json(indent=2)[:500] + "...")

# --------------------------------------------------------------
# Your Turn: Try extracting different document types
# --------------------------------------------------------------
"""
1. Try extracting a different document type (HTML, DOCX, etc.)
2. Explore the document structure to understand the standardized model
3. Consider how you would extract specific sections for your RAG application
"""
