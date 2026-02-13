import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from dotenv import load_dotenv
from src.ingestion import EarthquakeLoader, TextCleaner, TextSplitter
from src.vectorizer import EmbeddingModel, VectorStoreManager
from src.retrieval import Retriever
from src.rag import RAGChain

#os.environ["GROQ_API_KEY"] = ""


def load_earthquake_data(file_path: str):
    """Load and preprocess earthquake data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Earthquake file not found: {file_path}")

    loader = EarthquakeLoader()
    cleaner = TextCleaner()

    events = loader.load_txt(file_path)

    for event in events:
        event.page_content = cleaner.clean(event.page_content)

    return events


def build_vector_store(chunks):
    """Build the vector store index."""
    try:
        embedding_model = EmbeddingModel()
        vector_store = VectorStoreManager(embedding_model)
        vector_store.create_index(chunks)
        return vector_store
    except Exception as ex:
        raise RuntimeError(
            f"Failed to create vector index. Check API keys. Details: {ex}"
        )


def main():
    load_dotenv()

    # Correct project_root inside main
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(project_root, "..", "data")
    file_earthquakes = os.path.join(data_folder, "query.txt")

    # 1. Load data
    events = load_earthquake_data(file_earthquakes)

    # 2. Chunking
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(events)

    # 3. Vector store
    vector_store = build_vector_store(chunks)

    # 4. RAG chain
    retriever = Retriever(vector_store_manager=vector_store)
    rag_chain = RAGChain(retriever=retriever)

    # 5. CLI loop
    while True:
        question = input("\nEnter your question about earthquakes: ").strip()

        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = rag_chain.answer(question)

        print("\n>> RESPONSE:")
        print(result.get("risposta", "[No response]"))




if __name__ == "__main__":
    main()