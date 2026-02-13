import pytest
import os
from src.ingestion import DocumentLoader

def test_load_markdown_file(tmp_path):
    # Setup
    file_content = "# Test Document\n\nThis is a test."
    file_path = tmp_path / "test.md"
    file_path.write_text(file_content, encoding="utf-8")

    # Execution
    loader = DocumentLoader()
    docs = loader.load_file(str(file_path))

    # Verification
    assert len(docs) == 1
    assert docs[0].page_content == file_content
    assert docs[0].metadata["source"] == str(file_path)

def test_load_nonexistent_file():
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_file("nonexistent.md")

from src.ingestion import TextCleaner

def test_clean_text():
    cleaner = TextCleaner()
    raw_text = "  This is   a   test.  \n\n\n  New line.  "
    expected_text = "This is a test.\nNew line."
    assert cleaner.clean(raw_text) == expected_text

def test_clean_empty_text():
    cleaner = TextCleaner()
    assert cleaner.clean("") == ""

from src.ingestion import TextSplitter

def test_text_splitter():
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    # Create text longer than 100 chars
    long_text = "word " * 50  # 50 words, approx 250 chars
    
    chunks = splitter.split_text(long_text)
    
    assert len(chunks) > 1
    # Check that chunks are not exceeding limit too much (soft limit usually)
    # RecusriveCharacterTextSplitter usually strictly enforces if possible.
    assert len(chunks[0]) <= 100 
    
    # Check overlap (rough check)
    # If overlap works, the end of chunk 0 should match start of chunk 1 roughly
    
def test_text_splitter_documents():
    from langchain_core.documents import Document
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
    docs = [Document(page_content="word " * 50, metadata={"source": "test"})]
    
    split_docs = splitter.split_documents(docs)
    
    assert len(split_docs) > 1
    assert split_docs[0].metadata["source"] == "test"
