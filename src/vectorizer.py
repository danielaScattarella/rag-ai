from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ============================================================
# Embedding Model
# ============================================================

class EmbeddingModel:
    """
    Wrapper around a local HuggingFace embedding model.
    Provides vector embeddings for queries and documents.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding model.

        Args:
            model_name (str, optional): HuggingFace sentence-transformer model.
        """
        model_name = model_name or self.DEFAULT_MODEL

        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a single query string."""
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        return self.embeddings.embed_query(text)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        if not isinstance(documents, list) or not all(isinstance(d, str) for d in documents):
            raise TypeError("Documents must be a list of strings.")

        return self.embeddings.embed_documents(documents)


# ============================================================
# Vector Store Manager
# ============================================================

class VectorStoreManager:
    """
    Handles creation, management, and retrieval for a FAISS-based vector store.
    """

    def __init__(self, embedding_model: EmbeddingModel):
        """
        Initialize the manager.

        Args:
            embedding_model (EmbeddingModel): initialized embedding wrapper.
        """
        if not isinstance(embedding_model, EmbeddingModel):
            raise TypeError("embedding_model must be an instance of EmbeddingModel.")

        self.embedding_model = embedding_model
        self.vector_store: Optional[FAISS] = None

    def create_index(self, documents: List[Document]) -> None:
        """
        Build a new FAISS index from the given documents.

        Args:
            documents (List[Document]): list of LangChain Document objects.
        """
        if not documents:
            raise ValueError("Cannot create vector store: document list is empty.")

        self.vector_store = FAISS.from_documents(
            documents,
            self.embedding_model.embeddings
        )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to the existing FAISS index.

        Args:
            documents (List[Document]): list of Document objects.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_index() first.")

        if not documents:
            raise ValueError("Document list cannot be empty.")

        self.vector_store.add_documents(documents)

    def get_retriever(self, k: int = 4):
        """
        Retrieve a LangChain retriever interface.

        Args:
            k (int): number of top documents to return.

        Returns:
            A retriever object.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_index() first.")

        if k <= 0:
            raise ValueError("k must be a positive integer.")

        return self.vector_store.as_retriever(search_kwargs={"k": k})
