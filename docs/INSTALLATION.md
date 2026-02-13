# Installation Guide

This guide provides detailed installation instructions for the Enterprise RAG System.

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

## System Requirements

### Minimum Requirements
- **Python**: 3.9 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 2GB free space
- **Internet**: Required for initial setup and API calls

### Supported Operating Systems
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 20.04+, Debian, CentOS)

### Required Accounts
- **Groq API**: Free tier available at [console.groq.com](https://console.groq.com)

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Install Python

**Windows:**
1. Download Python from [python.org](https://python.org)
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Click "Install Now"

**macOS:**
```bash
# Using Homebrew
brew install python@3.11
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# CentOS/RHEL
sudo yum install python3.11
```

#### Step 2: Verify Python Installation

```bash
python --version
# Should show: Python 3.9.x or higher
```

#### Step 3: Clone Repository

```bash
git clone https://github.com/yourusername/enterprise-rag-system.git
cd enterprise-rag-system
```

#### Step 4: Create Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt.

#### Step 5: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `langchain` - RAG framework
- `langchain-groq` - Groq integration
- `langchain-huggingface` - HuggingFace embeddings
- `sentence-transformers` - Embedding models
- `faiss-cpu` - Vector database
- `streamlit` - Web interface
- `pytest` - Testing framework
- `python-dotenv` - Environment variables

#### Step 6: Configure API Key

1. Get your Groq API key from [console.groq.com/keys](https://console.groq.com/keys)

2. Create `.env` file in project root:

**Windows:**
```powershell
echo GROQ_API_KEY=your_key_here > .env
```

**macOS/Linux:**
```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

3. Replace `your_key_here` with your actual API key

#### Step 7: Verify Installation

```bash
# Run tests
pytest tests/

# Start Streamlit app
streamlit run app.py
```

If successful, your browser will open to `http://localhost:8501`

---

### Method 2: Docker Installation (Advanced)

**Coming soon** - Docker support is planned for future releases.

---

## Troubleshooting

### Issue: "python: command not found"

**Solution:**
- **Windows**: Reinstall Python with "Add to PATH" checked
- **macOS/Linux**: Use `python3` instead of `python`

### Issue: "pip: command not found"

**Solution:**
```bash
# Windows
python -m pip install --upgrade pip

# macOS/Linux
python3 -m pip install --upgrade pip
```

### Issue: "Permission denied" when installing packages

**Solution:**
```bash
# Don't use sudo! Use virtual environment instead
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "ModuleNotFoundError: No module named 'streamlit'"

**Solution:**
1. Ensure virtual environment is activated
2. Reinstall dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "API key not valid"

**Solution:**
1. Check `.env` file exists in project root
2. Verify format: `GROQ_API_KEY=gsk_...`
3. No quotes around the key
4. Get new key from [console.groq.com/keys](https://console.groq.com/keys)

### Issue: "Out of memory" when loading documents

**Solution:**
1. Reduce chunk size in `app.py`:
```python
splitter = TextSplitter(chunk_size=300, chunk_overlap=30)
```

2. Process fewer documents at once
3. Increase system RAM

### Issue: Slow embedding generation

**Solution:**
This is normal on first run. HuggingFace downloads the model (~90MB) once.
Subsequent runs are much faster.

---

## Verification

### 1. Check Python Version
```bash
python --version
# Expected: Python 3.9.x or higher
```

### 2. Check Virtual Environment
```bash
# Should see (.venv) in prompt
which python  # macOS/Linux
where python  # Windows
# Should point to .venv directory
```

### 3. Check Installed Packages
```bash
pip list
# Should show: streamlit, langchain, faiss-cpu, etc.
```

### 4. Check API Key
```bash
# Windows
type .env

# macOS/Linux
cat .env

# Should show: GROQ_API_KEY=gsk_...
```

### 5. Run Tests
```bash
pytest tests/ -v
# All tests should pass
```

### 6. Test Streamlit App
```bash
streamlit run app.py
# Browser should open to http://localhost:8501
```

---

## Next Steps

After successful installation:

1. **Add your documents** to the `data/` folder
2. **Restart the app** to load new documents
3. **Try example queries** from the README
4. **Read the [Usage Guide](USAGE.md)** for detailed features

---

## Getting Help

If you encounter issues not covered here:

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Search [GitHub Issues](https://github.com/danielaScattarella/enterprise-rag-system/issues)
3. Create a new issue with:
   - Your OS and Python version
   - Full error message
   - Steps to reproduce

---

Your Earthquake RAG System is now ready to use.

The system is fully initialized and prepared to:

- Load and process INGV earthquake data
- Perform high‑precision seismic event retrieval
- Run LLM‑powered Q&A grounded ONLY in your earthquake dataset
- Provide fast, deterministic responses (FAISS + MiniLM + Groq)
- Maintain metadata for each seismic event (EventID, Magnitudo, Profondità, Località…)
- Support expert queries about magnitudo, profondità, aree geografiche e zone sismiche
