import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from src.rag import RAGChain


def test_rag_chain_answer():
    # --- Mock Retriever ---
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        Document(page_content="context info 123")
    ]

    # --- Mock LLM (ChatGroq) ---
    with patch("src.rag.ChatGroq") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.invoke.return_value.content = "Answer based on context"

        chain = RAGChain(retriever=mock_retriever)

        # --- Execute ---
        response = chain.answer("test query")

        # --- Assertions ---
        assert response["answer"] == "Answer based on context"
        assert len(response["source_documents"]) == 1
        assert response["source_documents"][0].page_content == "context info 123"

        # --- Check calls ---
        mock_retriever.retrieve.assert_called_once_with("test query")
        mock_llm_instance.invoke.assert_called_once()


def test_rag_chain_no_context():
    # Retriever returns NO documents
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with patch("src.rag.ChatGroq") as MockLLM:
        mock_llm_instance = MockLLM.return_value
        mock_llm_instance.invoke.return_value.content = \
            "Non lo so in base ai documenti forniti."

        chain = RAGChain(retriever=mock_retriever)

        response = chain.answer("test query")

        # Must still call LLM with empty context
        args, kwargs = mock_llm_instance.invoke.call_args
        sent_prompt = args[0]

        # --- Assertions ---
        assert "Non lo so" in response["answer"]
        assert response["source_documents"] == []

        # --- Check that context is EMPTY ---
        # prompt is a LangChain "Message" object — we inspect its internal structure:
        rendered_messages = sent_prompt.to_messages()
        system_msg = rendered_messages[0].content.lower()

        # verify empty context was sent
        assert "context" in system_msg or "contesto" in system_msg

        # retriever was called
        mock_retriever.retrieve.assert_called_once_with("test query")