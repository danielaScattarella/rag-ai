# Troubleshooting Guide

Common issues and solutions for the Enterprise RAG System.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [API Issues](#api-issues)
- [Document Loading Issues](#document-loading-issues)

## Installation Issues

### Python Version Error

**Error:**
```
ERROR: This package requires Python 3.9 or higher
```

**Solution:**
```bash
# Check Python version
python --version

# Install Python 3.11 (recommended)
# Download from python.org
```

### Virtual Environment Not Activating

**Error:**
```
'activate' is not recognized as an internal or external command
```

**Solution (Windows):**
```powershell
# Use full path
.venv\Scripts\activate.bat

# Or use PowerShell
.venv\Scripts\Activate.ps1

# If PowerShell blocked, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Package Installation Fails

**Error:**
```
ERROR: Could not find a version that satisfies the requirement...
```

**Solution:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Then install requirements
pip install -r requirements.txt

# If still fails, install individually
pip install streamlit
pip install langchain
# etc.
```

---

## Runtime Errors

### API Key Not Found

**Error:**
```
groq.GroqError: The api_key client option must be set
```

**Solution:**
1. Create `.env` file in project root
2. Add: `GROQ_API_KEY=your_key_here`
3. Restart the application
4. Verify file exists: `cat .env` (macOS/Linux) or `type .env` (Windows)

### API Key Invalid

**Error:**
```
Error code: 401 - {'error': {'message': 'Invalid API key'}}
```

**Solution:**
1. Get new key from [console.groq.com/keys](https://console.groq.com/keys)
2. Update `.env` file
3. Ensure no quotes around key: `GROQ_API_KEY=gsk_...`
4. Restart application

### Model Not Found

**Error:**
```
Error code: 404 - {'error': {'message': 'The model `xyz` does not exist'}}
```

**Solution:**
Edit `src/rag.py`:
```python
# Use valid model name
model_name="llama-3.3-70b-versatile"
```

Valid models:
- `llama-3.3-70b-versatile`
- `llama-3.1-70b-versatile`
- `mixtral-8x7b-32768`

### Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Verify activation (should see .venv)
which python  # macOS/Linux
where python  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

---

## Performance Issues

### Slow First Run

**Symptom:** First run takes 2-3 minutes

**Explanation:** Normal! HuggingFace downloads embedding model (~90MB) on first run.

**Solution:** Wait for download to complete. Subsequent runs are much faster.

### Out of Memory

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
1. Reduce chunk size in `app.py`:
```python
splitter = TextSplitter(chunk_size=300, chunk_overlap=30)
```

2. Process fewer documents
3. Increase system RAM
4. Close other applications

### Slow Response Time

**Symptom:** Queries take >10 seconds

**Possible Causes:**
1. Too many chunks retrieved
2. Large documents
3. Slow internet connection

**Solutions:**
```python
# Reduce retrieval chunks (src/retrieval.py)
def retrieve(self, query: str, k: int = 4):  # Reduce from 8

# Reduce chunk size (app.py)
splitter = TextSplitter(chunk_size=300, chunk_overlap=30)
```

### Streamlit Keeps Reloading

**Symptom:** App reloads constantly

**Cause:** File watcher detecting changes

**Solution:**
```bash
# Disable file watcher
streamlit run app.py --server.fileWatcherType none
```

---

## API Issues

### Rate Limit Exceeded

**Error:**
```
Error code: 429 - {'error': {'message': 'Rate limit exceeded'}}
```

**Solution:**
1. Wait 60 seconds
2. Reduce query frequency
3. Upgrade Groq plan
4. Use caching for repeated queries

### Quota Exceeded

**Error:**
```
Error code: 429 - {'error': {'message': 'You exceeded your current quota'}}
```

**Solution:**
1. Check usage at [console.groq.com](https://console.groq.com)
2. Wait for quota reset (daily/monthly)
3. Upgrade to paid plan
4. Use different API key

### Connection Timeout

**Error:**
```
requests.exceptions.ConnectionError: Connection timeout
```

**Solution:**
1. Check internet connection
2. Check firewall settings
3. Try different network
4. Verify Groq API status

---

## Document Loading Issues

### No Documents Found

**Error:**
```
No documents found in data/ folder!
```

**Solution:**
1. Verify files are in `data/` folder
2. Check file extensions (`.md` or `,md`)
3. Ensure files are not empty
4. Restart Streamlit app

### File Not Loading

**Symptom:** Document count doesn't increase

**Possible Causes:**
1. Wrong file extension
2. File encoding issues
3. File too large

**Solutions:**
```bash
# Check file extension
ls data/  # macOS/Linux
dir data\  # Windows

# Verify file is readable
cat data/your_file.md  # macOS/Linux
type data\your_file.md  # Windows

# Check file size (should be <10MB)
ls -lh data/  # macOS/Linux
```

### Encoding Errors

**Error:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte...
```

**Solution:**
1. Convert file to UTF-8 encoding
2. Use text editor to save as UTF-8
3. Remove special characters

---

## UI Issues

### Streamlit Won't Start

**Error:**
```
streamlit: command not found
```

**Solution:**
```bash
# Activate virtual environment
.venv\Scripts\activate

# Verify streamlit installed
pip list | grep streamlit  # macOS/Linux
pip list | findstr streamlit  # Windows

# Reinstall if missing
pip install streamlit
```

### Port Already in Use

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process
# macOS/Linux
lsof -ti:8501 | xargs kill -9

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Browser Doesn't Open

**Symptom:** App starts but browser doesn't open

**Solution:**
1. Manually open: `http://localhost:8501`
2. Check firewall settings
3. Try different browser
4. Disable browser auto-open:
```bash
streamlit run app.py --server.headless true
```

---

## Testing Issues

### Tests Failing

**Error:**
```
FAILED tests/test_rag.py::test_answer
```

**Solution:**
1. Check if `.env` file exists (tests need it)
2. Verify all dependencies installed
3. Run tests with verbose output:
```bash
pytest tests/ -v
```

### Import Errors in Tests

**Error:**
```
ImportError: cannot import name 'RAGChain'
```

**Solution:**
```bash
# Ensure src/ is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # macOS/Linux
set PYTHONPATH=%PYTHONPATH%;%CD%  # Windows

# Or install package in editable mode
pip install -e .
```

---

## Getting More Help

If your issue isn't covered here:

1. **Check logs:**
```bash
# Streamlit logs
streamlit run app.py --logger.level debug
```

2. **Search GitHub Issues:**
   - [Existing Issues](https://github.com/danielaScattarella/enterprise-rag-system/issues)

3. **Create New Issue:**
   - Include error message
   - Include steps to reproduce
   - Include environment details:
```bash
python --version
pip list
cat .env  # Remove actual API key!
```

4. **Community Support:**
   - [GitHub Discussions](https://github.com/danielaScattarella/enterprise-rag-system/discussions)

---

**Still stuck?** Create a detailed issue on GitHub with:
- Full error message
- Steps to reproduce
- Your environment (OS, Python version)
- What you've tried so far
