# RAG Transformer: Machine Learning Assistant

## Project Overview

### What is RAG Transformer?

RAG (Retrieval-Augmented Generation) Transformer is an intelligent AI assistant that combines advanced machine learning techniques to provide contextually rich, informative responses across multiple knowledge domains:

#### Key Capabilities:
- **Intelligent Knowledge Retrieval**: Searches through a diverse dataset of machine learning, science fiction, and cosmic content
- **Context-Aware Responses**: Generates answers that are not just accurate, but contextually relevant
- **Multidomain Intelligence**: Bridges insights from:
  1. Machine Learning Concepts
  2. Science Fiction Movies
  3. Cosmic and Astronomical Observations

### How It Works

The system uses a sophisticated two-step process:
1. **Retrieval**: Finds the most relevant documents for a given query
2. **Generation**: Creates a coherent, informative response using the retrieved context

### Technical Architecture
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Generation Model**: `google/flan-t5-base`
- **Indexing**: FAISS semantic search
- **Data Sources**: TMDB and NASA APIs

### Unique Features
- Interactive query interface
- Adaptive learning
- Diverse knowledge base
- Real-time response generation

## Prerequisites
- Python 3.8+
- Git
- Terminal/Command Line

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/bniladridas/rag-transformer.git
cd rag-transformer
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Collect Datasets
```bash
python sci_fi_dataset_collector.py
```

### 5. Run RAG Transformer
```bash
python rag_pipeline.py
```

## Usage
- Type queries to interact
- Use 'help' for instructions
- Type 'exit' to quit

## Optional: Generate Knowledge Graph
```bash
python knowledge_graph_generator.py
```

## Troubleshooting
- Ensure all dependencies are installed
- Check Python version compatibility
- Verify API keys in .env file