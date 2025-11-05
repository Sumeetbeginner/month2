# Interactive RAG Demo

A comprehensive Streamlit-based demonstration of Retrieval-Augmented Generation (RAG) pipeline, showcasing the complete workflow from data ingestion to question answering.

## 🚀 Features

### Core RAG Pipeline
- **Data Ingestion**: Upload PDF files or paste raw text content
- **Text Chunking**: Intelligent text splitting with configurable chunk size and overlap
- **Embeddings**: Multiple sentence transformer models for text vectorization
- **Vector Storage**: Persistent ChromaDB for efficient similarity search
- **Question Answering**: Integration with Google's Gemini LLM for contextual responses
- **Citation System**: Automatic source attribution and chunk references

### Interactive UI Components
- **Visual Flow Diagram**: Graphviz-powered RAG pipeline visualization
- **Real-time Configuration**: Adjustable parameters for chunking, embeddings, and retrieval
- **Debug Mode**: Inspect retrieved chunks, similarity scores, and LLM prompts
- **Vector DB Overview**: Monitor stored chunks and collection statistics
- **Session Management**: Persistent state across interactions

### Advanced Features
- **Multi-format Support**: Handles both PDF and plain text documents
- **Model Selection**: Choose between different embedding models and Gemini variants
- **Performance Monitoring**: Response time tracking and token counting
- **Error Handling**: Graceful fallbacks for API failures and data issues

## 🏗️ Architecture

```
Data Input → Chunking → Embedding → Vector DB → Query → Retrieval → LLM → Answer + Citations
    ↑                                                                    ↓
    └───────────────────── ChromaDB Persistence ────────────────────────┘
```

### Components

1. **Data Processing Layer**
   - PDF text extraction using PyPDF2
   - Intelligent text chunking with overlap
   - Metadata preservation and source tracking

2. **Embedding Layer**
   - Sentence Transformers for semantic encoding
   - Configurable model selection
   - Batch processing for efficiency

3. **Vector Database Layer**
   - ChromaDB for persistent vector storage
   - Similarity search with distance metrics
   - Collection management and reset functionality

4. **Retrieval & Generation Layer**
   - Semantic similarity matching
   - Context-aware prompt construction
   - Gemini API integration for answer generation

## 📋 Prerequisites

- Python 3.8+
- Google Gemini API key (for LLM functionality)
- Internet connection for model downloads and API calls

## 🛠️ Installation

1. **Clone or download the project files**
   ```bash
   # Files needed: app.py, requirements.txt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Gemini API key**
   ```bash
   # Option 1: Environment variable
   export GEMINI_API_KEY="your-api-key-here"

   # Option 2: Direct in code (not recommended for production)
   # Edit app.py line ~47 to set your API key
   ```

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Step-by-Step Guide

1. **Configure Settings** (Sidebar)
   - Select embedding model (all-MiniLM-L6-v2 or all-mpnet-base-v2)
   - Choose Gemini model variant
   - Adjust chunk size (200-1600 tokens) and overlap (0-400)
   - Set ChromaDB persistence directory

2. **Add Context Data**
   - **Upload Files**: Drag and drop PDF or TXT files
   - **Paste Text**: Directly input text in the text area
   - Click "Ingest → Chunk → Embed → Store"

3. **Monitor Processing**
   - Watch real-time progress in the status indicator
   - View chunk counts per document
   - Check vector database metrics

4. **Ask Questions**
   - Enter your question in the query field
   - Adjust top-K retrieval parameter (1-10 chunks)
   - Enable debug mode to see retrieved chunks
   - Click "Run RAG" to generate answer

5. **Review Results**
   - Read the generated answer
   - Examine citations with source attribution
   - Inspect retrieved chunks and similarity scores

## ⚙️ Configuration Options

### Embedding Models
- `sentence-transformers/all-MiniLM-L6-v2`: Fast, lightweight (384D)
- `sentence-transformers/all-mpnet-base-v2`: Higher quality (768D)

### Gemini Models
- `gemini-2.5-flash`: Latest fast model with good performance

### Chunking Parameters
- **Chunk Size**: 200-1600 tokens (recommended: 800)
- **Overlap**: 0-400 tokens (recommended: 120)
- **Tokenization**: Whitespace-based for robustness

### Vector Database
- **Persistence Directory**: Default `.chromadb`
- **Collection Name**: `rag_demo`
- **Reset Functionality**: Clear and recreate collection

## 📊 Dependencies

```
streamlit>=1.28.0          # Web UI framework
chromadb>=0.4.0            # Vector database
sentence-transformers>=2.2.0  # Embedding models
pypdf>=3.0.0               # PDF text extraction
google-generativeai>=0.3.0 # Gemini LLM API
tiktoken>=0.5.0            # Token counting
```

## 🔧 API Integration

### Gemini API Setup
1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set as environment variable or edit the code
3. The app handles API configuration automatically

### Error Handling
- Network timeouts and API failures
- Invalid file formats
- Empty or corrupted documents
- Missing API keys

## 🎯 Use Cases

- **Document Q&A**: Ask questions about uploaded PDFs
- **Knowledge Base**: Build searchable document collections
- **Research Assistant**: Extract insights from academic papers
- **Content Analysis**: Summarize and query large text corpora
- **Educational Tool**: Demonstrate RAG concepts interactively

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'X'"**
   - Ensure all requirements are installed: `pip install -r requirements.txt`

2. **"API key not found"**
   - Set `GEMINI_API_KEY` environment variable
   - Or edit the hardcoded key in `app.py` (temporary)

3. **"ChromaDB connection failed"**
   - Check write permissions in the persistence directory
   - Try resetting the vector database

4. **Slow embedding generation**
   - Switch to smaller model (all-MiniLM-L6-v2)
   - Reduce chunk size for faster processing

5. **Poor answer quality**
   - Increase top-K retrieval chunks
   - Try different embedding model
   - Ensure relevant context is ingested

### Performance Tips

- Use smaller embedding models for faster processing
- Adjust chunk size based on document type
- Increase overlap for better context continuity
- Monitor token limits for LLM calls

## 🤝 Contributing

This is a demonstration project. For improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Google Gemini](https://ai.google.dev/) for language model
- [PyPDF2](https://pypdf2.readthedocs.io/) for PDF processing

---

**Note**: This application is designed for demonstration and development purposes. For production use, implement proper security measures, error handling, and scalability considerations.
