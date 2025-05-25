"""
Data Extraction for AI Applications: Common Scenarios

This file demonstrates the most typical data extraction patterns you'll encounter
when preparing data for vector databases using the Docling library.
"""

import os
from docling.document_converter import DocumentConverter


def extract_pdf_document(url: str):
    """Extract text from a PDF document."""
    doc = DocumentConverter().convert(url).document
    return doc.export_to_text()


def extract_web_page(url: str):
    """Extract text from a web page."""
    doc = DocumentConverter().convert(url).document
    return doc.export_to_text()


def extract_local_document(file_path: str):
    """Extract text from a local document."""
    doc = DocumentConverter().convert(file_path).document
    return doc.export_to_text()


def extract_structured_data(file_path: str):
    """Extract text from structured data (CSV, tables)."""
    doc = DocumentConverter().convert(file_path).document
    return doc.export_to_text()


def batch_extract(sources: list[str]):
    """Extract text from multiple sources in batch."""
    results = []
    for source in sources:
        try:
            doc = DocumentConverter().convert(source).document
            text = doc.export_to_text()
            results.append((source, text))
        except Exception as e:
            results.append((source, f"Error: {str(e)}"))
    return results


def main():
    # 1. PDF Research Papers
    pdf_text = extract_pdf_document("https://arxiv.org/pdf/2408.09869")
    print(f"PDF extraction: {len(pdf_text)} characters")

    # 2. Web Pages
    web_text = extract_web_page("https://python.org")
    print(f"Web extraction: {len(web_text)} characters")

    # 3. Local Documents
    sample_content = """
    # Company Policy Manual
    
    ## Remote Work Guidelines
    Employees may work remotely up to 3 days per week.
    
    ## Meeting Protocols
    All meetings must have an agenda shared 24 hours in advance.
    """
    with open("sample_doc.md", "w") as f:
        f.write(sample_content)

    local_text = extract_local_document("sample_doc.md")
    print(f"Local document extraction: {len(local_text)} characters")

    # 4. Structured Data
    csv_content = """Name,Role,Department
    Alice,Engineer,Tech
    Bob,Manager,Sales
    Carol,Analyst,Finance"""
    with open("sample_data.csv", "w") as f:
        f.write(csv_content)

    structured_text = extract_structured_data("sample_data.csv")
    print(f"Structured data extraction: {len(structured_text)} characters")

    # 5. Batch Processing
    sources = ["sample_doc.md", "sample_data.csv", "https://httpbin.org/html"]

    batch_results = batch_extract(sources)
    for source, result in batch_results:
        if isinstance(result, str) and result.startswith("Error"):
            print(f"Failed to extract {source}: {result}")
        else:
            print(f"Successfully extracted {source}: {len(result)} characters")

    # Cleanup
    try:
        os.remove("sample_doc.md")
        os.remove("sample_data.csv")
    except:
        pass


if __name__ == "__main__":
    main()
