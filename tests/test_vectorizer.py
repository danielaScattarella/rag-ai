import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.vectorizer import EmbeddingModel, VectorStoreManager

# ============================================================
# EMBEDDING MODEL TESTS (Earthquake Domain)
# ============================================================

def test_embedding_model_initialization():
    with patch("src.vectorizer.HuggingFaceEmbeddings") as MockEmbeddings:
        model = EmbeddingModel()
        MockEmbeddings.assert_called_once()


def test_embed_query_for_earthquake_question():
    with patch("src.vectorizer.HuggingFaceEmbeddings") as MockEmbeddings:
        mock_instance = MockEmbeddings.return_value
        mock_instance.embed_query.return_value = [0.12, 0.22, 0.32]

        model = EmbeddingModel()
        vec = model.embed_query("Terremoti con magnitudo maggiore di 5")

        assert vec == [0.12, 0.22, 0.32]
        mock_instance.embed_query.assert_called_with(
            "Terremoti con magnitudo maggiore di 5"
        )


def test_embed_documents_for_earthquake_chunks():
    with patch("src.vectorizer.HuggingFaceEmbeddings") as MockEmbeddings:
        mock_instance = MockEmbeddings.return_value
        mock_instance.embed_documents.return_value = [[0.1], [0.2]]

        model = EmbeddingModel()
        embeddings = model.embed_documents([
            "Event ID: 12345 - Magnitude 3.2",
            "Event ID: 99999 - Magnitude 4.8"
        ])

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1]
        mock_instance.embed_documents.assert_called_with([
            "Event ID: 12345 - Magnitude 3.2",
            "Event ID: 99999 - Magnitude 4.8"
        ])


# ============================================================
# VECTOR STORE MANAGER TESTS (Earthquake Domain)
# ============================================================

def test_vector_store_manager_create_index_for_earthquake_docs():
    with patch("src.vectorizer.FAISS") as MockFAISS:
        mock_embedding_model = MagicMock()
        manager = VectorStoreManager(embedding_model=mock_embedding_model)

        docs = [
            Document(
                page_content="Event ID: 123 - Magnitudo 4.1",
                metadata={"event_id": "123"}
            )
        ]

        manager.create_index(docs)

        MockFAISS.from_documents.assert_called_once_with(
            docs,
            mock_embedding_model.embeddings
        )


def test_vector_store_manager_add_earthquake_documents():
    with patch("src.vectorizer.FAISS") as MockFAISS:
        mock_embedding_model = MagicMock()
        manager = VectorStoreManager(embedding_model=mock_embedding_model)

        # Simulate existing FAISS vector index
        manager.vector_store = MockFAISS.return_value

        docs = [
            Document(
                page_content="Event ID: 777 - Depth 15 km",
                metadata={"event_id": "777"}
            )
        ]

        manager.add_documents(docs)
        manager.vector_store.add_documents.assert_called_once_with(docs)


def test_vector_store_metadata_preserved_for_earthquake_events():
    with patch("src.vectorizer.FAISS") as MockFAISS:
        mock_embedding_model = MagicMock()
        manager = VectorStoreManager(embedding_model=mock_embedding_model)

        docs = [
            Document(
                page_content="Chunk1",
                metadata={"event_id": "111", "latitude": 40.12}
            ),
            Document(
                page_content="Chunk2",
                metadata={"event_id": "222", "latitude": 38.90}
            )
        ]

        manager.create_index(docs)

        args, _ = MockFAISS.from_documents.call_args
        passed_docs = args[0]

        assert passed_docs[0].metadata["event_id"] == "111"
        assert passed_docs[0].metadata["latitude"] == 40.12
        assert passed_docs[1].metadata["event_id"] == "222"
        assert passed_docs[1].metadata["latitude"] == 38.90