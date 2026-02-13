EARTHQUAKE RAG SYSTEM
=====================

A production-ready Retrieval-Augmented Generation (RAG) system designed for analyzing INGV earthquake data.  
You can ask natural-language questions about Italian seismic events and receive grounded, deterministic answers sourced directly from the INGV catalog.

No hallucinations.  
If the answer is not present in the data, the system will respond:  
"Non lo so in base ai documenti forniti."

------------------------------------------------------------
KEY FEATURES
------------------------------------------------------------
- Earthquake-specific RAG system for INGV TXT/CSV data
- Fast semantic retrieval using HuggingFace MiniLM embeddings
- FAISS vector search for millisecond similarity search
- Groq Llama 3.3 70B (temperature 0) for deterministic answers
- Fully grounded: answers ONLY from the earthquake dataset
- Beautiful Streamlit chat interface
- Automatic ingestion of all earthquake files in /data/
- Source attribution: EventID, magnitude, depth, location, time

------------------------------------------------------------
PREREQUISITES
------------------------------------------------------------
- Python 3.9+
- Groq API Key (get it at https://console.groq.com)
- INGV earthquake datasets (TXT or CSV)
- Minimum 4 GB RAM

------------------------------------------------------------
INSTALLATION
------------------------------------------------------------

1. Clone the repository
-----------------------
git clone https://github.com/yourusername/earthquake-rag-system.git
cd earthquake-rag-system

2. Create a virtual environment
-------------------------------
python -m venv .venv
.venv\Scripts\activate        (Windows)

python3 -m venv .venv
source .venv/bin/activate     (macOS/Linux)

3. Install dependencies
-----------------------
pip install -r requirements.txt

4. Add your API key
-------------------
Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key_here

------------------------------------------------------------
USAGE
------------------------------------------------------------

Start the Streamlit web app:
----------------------------
streamlit run app.py

Then open:
http://localhost:8501

Available features:
- Ask questions about earthquakes (magnitude, depth, region...)
- View event sources returned by retrieval
- Clear chat history
- See loaded INGV files and chunk count

Command-line mode:
------------------
python src/main.py

------------------------------------------------------------
SUPPORTED EARTHQUAKE DATA FORMATS
------------------------------------------------------------

Place your INGV datasets in /data/

Example accepted TXT/CSV rows:

EventID|Time|Latitude|Longitude|Depth/Km|MagType|Magnitude|EventLocationName|EventType
41671442|2025-02-11T00:11:53.840000|42.8622|11.751|10.1|ML|0.9|4 km SW Radicofani (SI)|earthquake

The system supports:
- pipe-delimited .txt
- comma-delimited .csv


------------------------------------------------------------
CONFIGURATION
------------------------------------------------------------

Retrieval (src/retrieval.py):
k = 8   # number of chunks retrieved

Chunking (app.py or main.py):
chunk_size = 400
chunk_overlap = 40

RAG model (src/rag.py):
model_name = "llama-3.3-70b-versatile"

------------------------------------------------------------
TESTING
------------------------------------------------------------

Run all tests:
pytest tests/

With coverage:
pytest tests/ --cov=src

------------------------------------------------------------
PERFORMANCE
------------------------------------------------------------

Typical performance on a mid-range laptop:

Embedding: 40–60 ms  
FAISS retrieval: 8–15 ms  
LLM (Groq): 1.8–2.5 seconds  
End-to-end: 2–3 seconds per question

------------------------------------------------------------
EXAMPLE QUERIES
------------------------------------------------------------

"Quali terremoti recenti ci sono in Sicilia?"
"Eventi profondi nel Tirreno Meridionale?"
"Terremoti con magnitudo superiore a 5?"
"Mostrami i terremoti vicino ai Campi Flegrei."

------------------------------------------------------------
HOW IT WORKS
------------------------------------------------------------

1. Loads INGV TXT/CSV earthquake data
2. Cleans and normalizes text
3. Splits events into structured chunks
4. Embeds chunks using MiniLM
5. Indexes them using FAISS
6. Retrieves top-k relevant chunks
7. Llama 3.3 70B generates grounded answer
8. Returns EventID + source chunks

------------------------------------------------------------
TROUBLESHOOTING
------------------------------------------------------------

"No earthquake data found"
- Add .txt or .csv files to /data/

"API key invalid"
- Check .env file
- Validate key at Groq console

"Poor answers"
- Increase k in retrieval
- Increase chunk size
- Add more INGV datasets

------------------------------------------------------------
LICENSE
------------------------------------------------------------
MIT License

------------------------------------------------------------
CONTACT
------------------------------------------------------------
Open issues or discussions on GitHub.

------------------------------------------------------------
THANK YOU
------------------------------------------------------------
If this project helps you, please star the repository!