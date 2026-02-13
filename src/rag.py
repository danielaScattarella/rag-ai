from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from src.retrieval import Retriever
from src.prompts import create_rag_prompt_template


class RAGChain:
    

    def __init__(self, retriever: Retriever, llm_model: str = "llama-3.3-70b-versatile"):
        
        self.retriever = retriever
        self.llm = ChatGroq(model=llm_model, temperature=0)
        self.prompt_template = create_rag_prompt_template()

    def answer(self, question: str) -> Dict[str, Any]:
        

        # 1. Retrieve relevant documents
        documents: List[Document] = self.retriever.retrieve(question)

        # 2. Build context text
        context_text = "\n\n".join(doc.page_content for doc in documents)

        # 3. Construct prompt using template
        prompt_messages = self.prompt_template.invoke({
            "context": context_text,
            "question": question
        })

        # --- Logging / Observability ---
        print("\n--- [OBSERVABILITY] FINAL PROMPT SENT TO MODEL ---")
        for message in prompt_messages.to_messages():
            print(f"[{message.type.upper()}]: {message.content}")
        print("---------------------------------------------------\n")

        # 4. LLM inference
        model_response = self.llm.invoke(prompt_messages)

        # Normalize possible heterogeneous outputs
        content = model_response.content
        if isinstance(content, list):
            # Some models return structured output
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        elif not isinstance(content, str):
            content = str(content)

        # --- Logging / Observability ---
        print("\n--- [OBSERVABILITY] RAW MODEL RESPONSE ---")
        print(model_response.content)
        print("---------------------------------------------------\n")

        # 5. Return structured result
        return {
            "answer": content,
            "source_documents": documents,
            "question": question,
            "generated_prompt": prompt_messages
        }