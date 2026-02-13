import pytest
from unittest.mock import MagicMock, patch
from src.rag import RAGChain
from langchain_core.documents import Document

def test_fallback_when_retrieval_empty():
    """
    When no earthquake events match the query,
    the RAG system must return the safe fallback:
    'Non lo so in base ai documenti forniti.'
    This matches the Earthquake RAG rules.
    """

    # --- Mock retriever: NO EARTHQUAKES FOUND ---
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []   # No INGV events

    # --- Mock LLM (ChatGroq) ---
    with patch("src.rag.ChatGroq") as MockChat:
        mock_llm = MockChat.return_value

        # Expected fallback according to your Earthquake RAG system rules
        mock_llm.invoke.return_value.content = (
            "Non lo so in base ai documenti forniti."
        )

        chain = RAGChain(retriever=mock_retriever)

        # Query example for earthquakes
        response = chain.answer("Ci sono terremoti con magnitudo superiore a 7?")

        # --- Validate answer ---
        assert "Non lo so" in response["answer"]
        assert response["source_documents"] == []

        # --- Verify retriever call ---
        mock_retriever.retrieve.assert_called_once_with(
            "Ci sono terremoti con magnitudo superiore a 7?"
        )

        # --- Verify LLM was invoked ---
        mock_llm.invoke.assert_called_once()

        # --- Inspect the constructed prompt ---
        prompt_obj = mock_llm.invoke.call_args[0][0]

        # Convert to LangChain message objects
        messages = prompt_obj.to_messages()
        system_msg = messages[0].content.lower()

        # --- Ensure the CONTEXT is EMPTY ---
        # Because no earthquake events were found
        assert "context" in system_msg
        assert "event id" not in system_msg
        assert "magnitude" not in system_msg
        assert "depth" not in system_msg
        assert "location" not in system_msg
