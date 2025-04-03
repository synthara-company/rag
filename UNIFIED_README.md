# Unified RAG System

![RAG Transformer](https://img.shields.io/badge/RAG-Transformer-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive Retrieval-Augmented Generation system that combines hybrid retrieval, multi-modal capabilities, personalization, and vector database integration.

## Features

- **Hybrid Retrieval**: Combines dense and sparse retrieval methods for better document retrieval
- **Multi-Modal Capabilities**: Handles both text and image inputs and queries
- **Personalization**: Adapts responses based on user preferences and history
- **Vector Database Integration**: Uses vector databases for efficient storage and retrieval
- **Advanced Prompting**: Uses few-shot examples and domain-specific templates
- **Response Evaluation**: Assesses and improves the quality of generated responses
- **Multiple Model Support**: Integrates with Hugging Face, OpenAI, and Anthropic models
- **Knowledge Graph Visualization**: Creates visual representations of relationships between concepts
- **Web Interface**: User-friendly interface with analytics and knowledge base exploration

## Architecture

The Unified RAG System integrates multiple components:

1. **Hybrid Retriever**: Combines BM25, TF-IDF, and dense embeddings for better retrieval
2. **Multi-Modal RAG**: Handles text and image inputs using CLIP and other models
3. **Personalization Engine**: Adapts responses based on user profiles and history
4. **Vector Database**: Efficiently stores and retrieves document embeddings
5. **Advanced Prompting**: Uses templates and few-shot examples for better responses
6. **Response Evaluator**: Assesses and improves the quality of generated responses
7. **Model Factory**: Creates language model instances from different providers

## Installation

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-transformer.git
   cd rag-transformer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up API keys:
   Create a `.env` file in the project root with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key  # Optional
   ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional
   PINECONE_API_KEY=your_pinecone_api_key  # Optional
   PINECONE_ENVIRONMENT=your_pinecone_environment  # Optional
   ```

## Usage

### Configuration

The system can be configured using the `config.json` file:

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "vision_model": "openai/clip-vit-base-patch32",
  "generator_model": "google/flan-t5-small",
  "generator_model_type": "huggingface",
  "generator_model_name": "google/flan-t5-small",
  "use_bm25": true,
  "use_tfidf": true,
  "retrieval_alpha": 0.7,
  "profiles_dir": "user_profiles",
  "vector_db_type": "chroma",
  "vector_db_config": {
    "persist_directory": "vector_db"
  },
  "use_hybrid_retrieval": true,
  "use_multimodal": true,
  "use_personalization": true,
  "use_vector_db": true
}
```

### Web Interface

Launch the web interface:
```bash
streamlit run unified_web_interface.py
```

### Python API

```python
from unified_rag_system import UnifiedRAGSystem

# Initialize the system
rag_system = UnifiedRAGSystem(
    config_path="config.json",
    use_hybrid_retrieval=True,
    use_multimodal=True,
    use_personalization=True,
    use_vector_db=True,
    vector_db_type='chroma'
)

# Add documents
documents = [
    "Machine learning is a subset of artificial intelligence...",
    "Deep learning uses neural networks with multiple layers..."
]
sources = ["ML", "ML"]
metadata = [{"author": "Author 1"}, {"author": "Author 2"}]

rag_system.add_documents(documents, sources, metadata)

# Process a query
user_id = "user123"
query = "What is deep learning?"

response = rag_system.process_query(
    query,
    user_id=user_id,
    include_images=True,
    text_top_k=5,
    image_top_k=3
)

print(f"Response: {response['response']}")
```

## Components

### Hybrid Retrieval

The hybrid retrieval system combines dense and sparse retrieval methods:

```python
from hybrid_retrieval import HybridRetriever

retriever = HybridRetriever(
    embedding_model="all-MiniLM-L6-v2",
    use_bm25=True,
    use_tfidf=True,
    alpha=0.7
)

# Add documents
retriever.add_documents(documents, sources, metadata)

# Retrieve documents
docs, sources, metadata, scores, details = retriever.retrieve(query, top_k=5)
```

### Multi-Modal RAG

The multi-modal RAG system handles both text and image inputs:

```python
from multimodal_rag import MultiModalRAG

mm_rag = MultiModalRAG(
    text_embedding_model="all-MiniLM-L6-v2",
    vision_model="openai/clip-vit-base-patch32",
    generator_model="google/flan-t5-small"
)

# Add text documents
mm_rag.add_text_documents(documents, sources, metadata)

# Add images
from PIL import Image
images = [Image.open("image1.jpg"), Image.open("image2.jpg")]
captions = ["Caption 1", "Caption 2"]
mm_rag.add_images(images, captions)

# Generate response
response = mm_rag.generate_response(query)
```

### Personalization

The personalization engine adapts responses based on user preferences:

```python
from personalization import PersonalizationEngine

personalization = PersonalizationEngine(
    embedding_model="all-MiniLM-L6-v2",
    profiles_dir="user_profiles"
)

# Get user profile
user_id = "user123"
profile = personalization.get_profile(user_id)

# Update profile from query
personalization.update_profile_from_query(user_id, query, response)

# Personalize retrieval
personalized_docs, personalized_sources, personalized_scores = personalization.personalize_retrieval(
    user_id, docs, sources, scores, top_k=5
)

# Personalize prompt
personalized_prompt = personalization.personalize_prompt(user_id, prompt)
```

### Vector Database Integration

The vector database integration provides efficient storage and retrieval:

```python
from vector_db_integration import create_vector_db

# Create vector database
vector_db = create_vector_db(
    'chroma',
    persist_directory="vector_db",
    embedding_model="all-MiniLM-L6-v2"
)

# Add documents
doc_ids = vector_db.add_documents(documents, metadata, collection_name="default")

# Search
docs, metadata, scores = vector_db.search(query, top_k=5, collection_name="default")
```

## Advanced Features

### Knowledge Graph Visualization

```python
from knowledge_graph_visualizer import KnowledgeGraphBuilder, KnowledgeGraphVisualizer

# Build knowledge graph
builder = KnowledgeGraphBuilder(use_spacy=True, embedding_model="all-MiniLM-L6-v2")
graph = builder.build_graph_from_documents(documents, sources)

# Prune graph
pruned_graph = builder.prune_graph(min_edge_weight=0.5, min_node_frequency=2, max_nodes=100)

# Visualize graph
visualizer = KnowledgeGraphVisualizer()
fig = visualizer.visualize_matplotlib(pruned_graph, "knowledge_graph.png", "Knowledge Graph")
```

### Model Integrations

```python
from model_integrations import ModelFactory

# Create model
model_factory = ModelFactory()
model = model_factory.create_model('openai', 'gpt-3.5-turbo')

# Generate text
result = model.generate("What is machine learning?")
print(result['text'])
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the transformer models
- OpenAI and Anthropic for their language models
- FAISS, ChromaDB, Pinecone, and Qdrant for vector search capabilities
- Streamlit for the web interface
- The open-source community for various libraries and tools
