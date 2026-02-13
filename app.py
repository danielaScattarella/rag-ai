import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.ingestion import EarthquakeLoader, TextCleaner, TextSplitter
from src.vectorizer import EmbeddingModel, VectorStoreManager
from src.retrieval import Retriever
from src.rag import RAGChain

# Page config
st.set_page_config(
    page_title="Earthquake RAG System",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS theme
st.markdown("""
<style>
.main { background: #f7f9fc; }
.stApp { background: #f7f9fc; }
.chat-container {
    background: white; border-radius: 15px; padding: 20px; margin: 10px 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.user-message {
    background: #0057d9; color: white; padding: 15px; border-radius: 15px;
    margin: 10px 0; max-width: 80%; margin-left: auto;
}
.assistant-message {
    background: #e8eaf6; padding: 15px; border-radius: 15px;
    margin: 10px 0; max-width: 80%;
}
.source-box {
    background: #eef2ff; padding: 10px; border-radius: 10px; margin: 6px 0;
    border-left: 4px solid #0057d9; color: #1a1a1a; font-size: 14px;
}
h1 { color: #001A43; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system():
    """Initialize the Earthquake RAG System."""
    load_dotenv()

    with st.spinner("🌋 Loading earthquake catalog and building index..."):
        loader = EarthquakeLoader()
        cleaner = TextCleaner()
        splitter = TextSplitter(chunk_size=400, chunk_overlap=40)

        data_dir = os.path.join(os.path.dirname(__file__), "data")
        all_chunks = []
        loaded_files = []

        # Load earthquake datasets (TXT or CSV)
        for filename in os.listdir(data_dir):
            if filename.lower().endswith(".txt") or filename.lower().endswith(".csv"):
                file_path = os.path.join(data_dir, filename)
                try:
                    raw_docs = loader.load_txt(file_path)
                    for doc in raw_docs:
                        doc.page_content = cleaner.clean(doc.page_content)
                    file_chunks = splitter.split_documents(raw_docs)
                    all_chunks.extend(file_chunks)
                    loaded_files.append(filename)
                except Exception as e:
                    st.warning(f"Could not load {filename}: {str(e)}")

        if not all_chunks:
            raise Exception("No earthquake data found in /data folder.")

        # Build vector index
        embedding_model = EmbeddingModel()
        manager = VectorStoreManager(embedding_model)
        manager.create_index(all_chunks)

        retriever = Retriever(vector_store_manager=manager)

        # Save info to session state
        st.session_state.loaded_files = loaded_files
        st.session_state.total_chunks = len(all_chunks)

        return RAGChain(retriever=retriever)


def main():

    st.title("🌋 Earthquake RAG System")
    st.markdown("### Ask questions about INGV Earthquake Events")

    # Sidebar
    with st.sidebar:
        st.header("📊 System Info")
        st.info("""
Powered by:
- INGV Earthquake Catalog
- Groq Llama 3.3 70B
- MiniLM Embeddings
- FAISS Vector Search
        """)

        st.header("📁 Loaded Earthquake Files")
        if "loaded_files" in st.session_state:
            st.success(f"{len(st.session_state.loaded_files)} files loaded")
            st.info(f"Total chunks: {st.session_state.total_chunks}")
            with st.expander("View loaded files"):
                for file in st.session_state.loaded_files:
                    st.write(f"✓ {file}")
        else:
            st.info("Loading earthquake data...")

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Init chat memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Init RAG system
    try:
        rag_chain = initialize_rag_system()
        st.success("🌋 Earthquake RAG System Ready!")
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "sources" in message and message["sources"]:
                with st.expander("Event Sources"):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(f"""
<div class="source-box">
<strong>Event {i}</strong><br>
ID: {src['event_id']}<br>
<em>{src['content'][:200]}...</em>
</div>
                        """, unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask about earthquakes..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing earthquake data..."):
                try:
                    result = rag_chain.answer(prompt)
                    answer = result["answer"]
                    sources = result["source_documents"]

                    # Show answer
                    st.markdown(answer)

                    # Show sources
                    if sources:
                        with st.expander("Event Sources"):
                            for i, doc in enumerate(sources, 1):
                                st.markdown(f"""
<div class="source-box">
<strong>Event {i}</strong><br>
ID: {doc.metadata.get('event_id', 'unknown')}<br>
<em>{doc.page_content[:200]}...</em>
</div>
                                """, unsafe_allow_html=True)

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": [
                            {
                                "event_id": doc.metadata.get("event_id", "unknown"),
                                "content": doc.page_content
                            }
                            for doc in sources
                        ]
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()
