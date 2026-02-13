from typing import List, Optional
from langchain_core.documents import Document
from src.vectorizer import VectorStoreManager

class Retriever:
    def __init__(self, vector_store_manager: VectorStoreManager):
        """
        Initialize the Retriever.
        
        Args:
            vector_store_manager (VectorStoreManager): The managed vector store.
        """
        self.vector_store_manager = vector_store_manager

    def retrieve(self, query: str, k: int = 8) -> List[Document]:
        """
        Retrieve relevant documents for the query.
        
        Args:
            query (str): The search query.
            k (int): Number of documents to retrieve.
            
        Returns:
            List[Document]: Retrieved documents.
        """
        if not query or not query.strip():
            # Decide on behavior for empty query. Returning empty list is safest.
            return []
            
        retriever = self.vector_store_manager.get_retriever(k=k)
        return retriever.invoke(query)

    def retrieve_with_logs(self, query: str, k: int = 8):
        """
        Retrieve documents and return them with detailed logging info.
        
        Args:
            query (str): The search query.
            k (int): Number of documents.
            
        Returns:
            dict: Contains 'results' (documents) and 'logs' (list of dicts).
        """
        if not query or not query.strip():
            return {"results": [], "logs": []}
            
        # We need to access the vector store directly to get scores if possible,
        # but standard retriever.invoke() returns just docs.
        # To get scores, we might need similarity_search_with_score on the store.
        
        vector_store = self.vector_store_manager.vector_store
        if vector_store is None:
             raise ValueError("Vector store not initialized.")
             
        # Perform search with scores
        docs_and_scores = vector_store.similarity_search_with_score(query, k=k)
        
        results = []
        logs = []
        
        for i, (doc, score) in enumerate(docs_and_scores):
            results.append(doc)
            logs.append({
                "rank": i + 1,
                "content_snippet": doc.page_content[:100] + "...",
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score) # Lower is better for L2, Higher for Cosine usually (FAISS default depends)
            })
            
        return {"results": results, "logs": logs}
