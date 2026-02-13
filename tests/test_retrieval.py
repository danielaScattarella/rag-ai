import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.retrieval import Retriever


def test_retrieve_documents():
    # Mock vector store manager
    mock_manager = MagicMock()

    # Mock retriever returned by manager.get_retriever()
    mock_retriever = MagicMock()
    mock_manager.get_retriever.return_value = mock_retriever

    # Mock FAISS results
    expected_docs = [Document(page_content="Event: magnitude 3.5")]
    mock_retriever.invoke.return_value = expected_docs

    retriever = Retriever(vector_store_manager=mock_manager)

    # Execute
    results = retriever.retrieve("terremoti in Sicilia")

    # Verify results
    assert len(results) == 1
    assert results[0].page_content == "Event: magnitude 3.5"

    # Verify retriever interactions
    mock_manager.get_retriever.assert_called_once()
    mock_retriever.invoke.assert_called_with("terremoti in Sicilia")


def test_retrieve_empty_query_returns_empty_list():
    mock_manager = MagicMock()
    retriever = Retriever(vector_store_manager=mock_manager)

    results = retriever.retrieve("   ")

    assert results == []


def test_retrieve_with_logs_basic():
    # Mock FAISS vector store
    mock_manager = MagicMock()
    mock_vector_store = MagicMock()
    mock_manager.vector_store = mock_vector_store

    # Simulated FAISS return: (doc, score)
    doc1 = Document(page_content="Event ID: 111, Magnitudo 3.2",
                    metadata={"source": "file1.txt"})
    doc2 = Document(page_content="Event ID: 222, Magnitudo 4.1",
                    metadata={"source": "file2.txt"})

    mock_vector_store.similarity_search_with_score.return_value = [
        (doc1, 1.23),
        (doc2, 1.56)
    ]

    retriever = Retriever(vector_store_manager=mock_manager)

    result = retriever.retrieve_with_logs("terremoti profondi", k=2)

    # Validate structure
    assert "results" in result
    assert "logs" in result
    assert len(result["results"]) == 2
    assert len(result["logs"]) == 2

    # Validate first log
    log1 = result["logs"][0]
    assert log1["rank"] == 1
    assert "Event ID: 111" in log1["content_snippet"]
    assert log1["source"] == "file1.txt"
    assert isinstance(log1["score"], float)

    # Ensure FAISS was called
    mock_vector_store.similarity_search_with_score.assert_called_with(
        "terremoti profondi", k=2
    )


def test_retrieve_with_logs_empty_query():
    mock_manager = MagicMock()
    retriever = Retriever(vector_store_manager=mock_manager)

    result = retriever.retrieve_with_logs("   ")

    assert result["results"] == []
    assert result["logs"] == []


def test_retrieve_with_logs_vectorstore_not_initialized():
    mock_manager = MagicMock()
    mock_manager.vector_store = None

    retriever = Retriever(vector_store_manager=mock_manager)

    with pytest.raises(ValueError):
        retriever.retrieve_with_logs("terremoti nel Tirreno", k=3)