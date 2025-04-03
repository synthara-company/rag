import os
import json
import torch
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import Counter

# Load environment variables
load_dotenv()

class EnhancedRAGTransformer:
    def __init__(self,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 generator_model: str = 'google/flan-t5-small'):
        """
        Initialize Enhanced RAG Transformer with embedding and generation models

        Args:
            embedding_model (str): Model for creating document embeddings
            generator_model (str): Model for text generation
        """
        # Embedding Model for Retrieval
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise

        # Generator Model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model)
        except Exception as e:
            print(f"Error loading generator model: {e}")
            raise

        # Knowledge Base
        self.knowledge_base = []
        self.document_sources = []  # Track the source of each document
        self.index = None
        
        # Query history
        self.query_history = []
        self.response_history = []
        
        # Performance metrics
        self.retrieval_times = []
        self.generation_times = []

    def load_datasets(self, dataset_dir: str = os.path.join(os.path.dirname(__file__), 'datasets')):
        """
        Load datasets from JSON files in the specified directory

        Args:
            dataset_dir (str): Directory containing dataset JSON files
        """
        # Machine learning documents
        ml_documents = [
            "Machine learning is a subset of artificial intelligence that focuses on the use of data and algorithms to imitate the way that humans learn.",
            "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
            "Transformers are a type of neural network architecture used in NLP that can handle sequential data more effectively than previous models.",
            "Supervised learning involves training a model on labeled data, where the desired output is known.",
            "Unsupervised learning finds hidden patterns or intrinsic structures in input data without labeled responses.",
            "Reinforcement learning is about training an agent to make decisions by rewarding desired behaviors and punishing undesired ones.",
            "Neural networks are computing systems inspired by biological neural networks that can learn to perform tasks by considering examples.",
            "Feature extraction is a key process in machine learning where important characteristics are identified and selected from raw data.",
            "Overfitting occurs when a machine learning model learns the training data too well, including its noise and fluctuations."
        ]
        
        # Add ML documents to knowledge base
        for doc in ml_documents:
            self.knowledge_base.append(doc)
            self.document_sources.append("Machine Learning")

        # Load sci-fi movies dataset
        sci_fi_movies_path = os.path.join(dataset_dir, 'sci_fi_movies.json')
        cosmos_content_path = os.path.join(dataset_dir, 'cosmos_content.json')

        try:
            with open(sci_fi_movies_path, 'r') as f:
                sci_fi_movies = json.load(f)
                # Extract movie titles and overviews
                for movie in sci_fi_movies:
                    movie_doc = f"Science Fiction Movie: {movie['title']}. Release Date: {movie.get('release_date', 'Unknown')}. Overview: {movie['overview']}. Popularity: {movie.get('popularity', 'N/A')}. Genres: {', '.join(movie.get('genres', []))}"
                    self.knowledge_base.append(movie_doc)
                    self.document_sources.append("Science Fiction")
                print(f"Loaded {len(sci_fi_movies)} sci-fi movie documents")
        except FileNotFoundError:
            print(f"Sci-fi movies dataset not found at {sci_fi_movies_path}")

        try:
            with open(cosmos_content_path, 'r') as f:
                cosmos_content = json.load(f)
                # Extract cosmos content details
                for content in cosmos_content:
                    cosmos_doc = f"Cosmos Observation Date: {content.get('date', 'Unknown')}. Title: {content.get('title', 'No Title')}. Media Type: {content.get('media_type', 'Unknown')}. Explanation: {content.get('explanation', 'No details')}"
                    self.knowledge_base.append(cosmos_doc)
                    self.document_sources.append("Cosmos")
                print(f"Loaded {len(cosmos_content)} cosmos content documents")
        except FileNotFoundError:
            print(f"Cosmos content dataset not found at {cosmos_content_path}")

        # Create FAISS index
        self._create_index()
        
    def _create_index(self):
        """Create FAISS index from the knowledge base"""
        if not self.knowledge_base:
            print("Knowledge base is empty. Cannot create index.")
            return
            
        # Embed documents
        embeddings = self.embedding_model.encode(self.knowledge_base)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        print(f"Created FAISS index with {len(self.knowledge_base)} documents")

    def add_documents(self, documents: List[str], source: str = "Custom"):
        """
        Add documents to the knowledge base and update FAISS index

        Args:
            documents (List[str]): List of documents to add
            source (str): Source category of the documents
        """
        if not documents:
            return
            
        # Add documents to knowledge base
        for doc in documents:
            self.knowledge_base.append(doc)
            self.document_sources.append(source)
            
        # Recreate index
        self._create_index()

    def retrieve_context(self, query: str, top_k: int = 5) -> Tuple[List[str], List[str], List[float]]:
        """
        Retrieve most relevant documents for a given query

        Args:
            query (str): Input query
            top_k (int): Number of documents to retrieve

        Returns:
            Tuple[List[str], List[str], List[float]]: Retrieved documents, their sources, and relevance scores
        """
        start_time = time.time()
        
        # Handle very short or non-informative queries
        if len(query.split()) < 2:
            docs = self.knowledge_base[:min(top_k, len(self.knowledge_base))]
            sources = self.document_sources[:min(top_k, len(self.document_sources))]
            scores = [1.0] * len(docs)  # Default score
            
            self.retrieval_times.append(time.time() - start_time)
            return docs, sources, scores

        # Encode query
        query_embedding = self.embedding_model.encode([query])

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Convert distances to similarity scores (lower distance = higher similarity)
        max_dist = np.max(distances[0]) if distances[0].size > 0 else 1.0
        similarity_scores = [1.0 - (dist / max_dist) for dist in distances[0]]

        # Retrieve documents and their sources
        retrieved_docs = [self.knowledge_base[idx] for idx in indices[0]]
        retrieved_sources = [self.document_sources[idx] for idx in indices[0]]
        
        self.retrieval_times.append(time.time() - start_time)
        return retrieved_docs, retrieved_sources, similarity_scores

    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response using retrieved context with enhanced features

        Args:
            query (str): Input query

        Returns:
            Dict[str, Any]: Response data including text, sources, and metrics
        """
        # Add query to history
        self.query_history.append(query)
        
        try:
            # Handle greeting-like queries
            greetings = ['hi', 'hello', 'hey', 'greetings']
            if query.lower().split() and query.lower().split()[0] in greetings:
                response = "Hello! I'm an AI assistant with knowledge about machine learning, sci-fi movies, and cosmos. How can I help you today?"
                self.response_history.append(response)
                return {
                    "response": response,
                    "sources": ["System"],
                    "retrieval_time": 0,
                    "generation_time": 0,
                    "context_docs": []
                }

            # Simple question handling
            if "how are you" in query.lower():
                response = "I'm functioning well, thank you for asking! I'm here to help with information about machine learning, sci-fi movies, and cosmos."
                self.response_history.append(response)
                return {
                    "response": response,
                    "sources": ["System"],
                    "retrieval_time": 0,
                    "generation_time": 0,
                    "context_docs": []
                }

            # Retrieve context
            context_docs, sources, scores = self.retrieve_context(query)
            
            # Start generation timer
            gen_start_time = time.time()
            
            # Prepare input with more detailed instructions
            context = " ".join(context_docs)
            input_text = f"""Context: {context}

Query: {query}

Instructions:
1. Provide a concise and informative response based on the context.
2. Focus on directly answering the query.
3. If the context doesn't contain relevant information, say so.
4. Do not make up information.

Response:"""

            # Generate response with safer parameters
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = self.generator.generate(
                    **inputs,
                    max_length=150,
                    num_return_sequences=1,
                    do_sample=True,  # Use sampling for more diverse responses
                    temperature=0.7,  # Lower temperature for more focused responses
                    top_p=0.9,  # Nucleus sampling
                    no_repeat_ngram_size=3  # Avoid repeating the same phrases
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Record generation time
            generation_time = time.time() - gen_start_time
            self.generation_times.append(generation_time)
            
            # Get the most recent retrieval time
            retrieval_time = self.retrieval_times[-1] if self.retrieval_times else 0

            # Fallback for very generic queries
            if not response or len(response.split()) < 3:
                response = "I apologize, but I couldn't generate a specific response. Could you please rephrase your query about machine learning, sci-fi movies, or cosmos?"
            
            # Add response to history
            self.response_history.append(response)
            
            # Determine primary source category
            source_counter = Counter(sources)
            primary_source = source_counter.most_common(1)[0][0] if source_counter else "Unknown"
            
            return {
                "response": response,
                "sources": sources,
                "primary_source": primary_source,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "context_docs": context_docs,
                "similarity_scores": scores
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            response = "I encountered an error while processing your query. Please try a different question about machine learning, sci-fi movies, or cosmos."
            self.response_history.append(response)
            return {
                "response": response,
                "sources": ["Error"],
                "retrieval_time": 0,
                "generation_time": 0,
                "context_docs": []
            }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the RAG system

        Returns:
            Dict[str, Any]: Performance metrics
        """
        metrics = {
            "avg_retrieval_time": np.mean(self.retrieval_times) if self.retrieval_times else 0,
            "avg_generation_time": np.mean(self.generation_times) if self.generation_times else 0,
            "total_queries": len(self.query_history),
            "knowledge_base_size": len(self.knowledge_base),
            "source_distribution": Counter(self.document_sources)
        }
        return metrics
    
    def visualize_source_distribution(self) -> Figure:
        """
        Visualize the distribution of document sources in the knowledge base

        Returns:
            Figure: Matplotlib figure with the visualization
        """
        source_counts = Counter(self.document_sources)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sources = list(source_counts.keys())
        counts = list(source_counts.values())
        
        ax.bar(sources, counts, color=['blue', 'green', 'purple'])
        ax.set_title('Knowledge Base Source Distribution')
        ax.set_xlabel('Source')
        ax.set_ylabel('Number of Documents')
        
        for i, count in enumerate(counts):
            ax.text(i, count + 0.1, str(count), ha='center')
            
        plt.tight_layout()
        return fig

def main():
    # Initialize Enhanced RAG Transformer
    rag = EnhancedRAGTransformer()
    
    # Load datasets
    rag.load_datasets()
    
    # Interactive query loop
    print("Enhanced RAG Transformer - ML, Sci-Fi, and Cosmos Assistant")
    print("Type 'exit' to quit the program")
    print("Type 'help' for usage instructions")
    print("Type 'stats' to see performance metrics")

    while True:
        # Get user input
        query = input("\nEnter your query (or 'exit'/'help'/'stats'): ").strip()
        
        # Check for special commands
        if query.lower() == 'exit':
            print("Exiting Enhanced RAG Transformer. Goodbye!")
            break
            
        if query.lower() == 'help':
            print("\nThis is an Enhanced AI Assistant covering:")
            print("- Machine Learning concepts")
            print("- Science Fiction Movies")
            print("- Cosmos and Astronomy")
            print("\nSpecial commands:")
            print("- 'stats': View system performance metrics")
            print("- 'exit': Quit the program")
            print("- 'help': Show this help message")
            continue
            
        if query.lower() == 'stats':
            metrics = rag.get_performance_metrics()
            print("\n=== Performance Metrics ===")
            print(f"Total Queries: {metrics['total_queries']}")
            print(f"Knowledge Base Size: {metrics['knowledge_base_size']}")
            print(f"Average Retrieval Time: {metrics['avg_retrieval_time']:.4f} seconds")
            print(f"Average Generation Time: {metrics['avg_generation_time']:.4f} seconds")
            print("\nSource Distribution:")
            for source, count in metrics['source_distribution'].most_common():
                print(f"- {source}: {count} documents")
            continue
            
        # Check for empty query
        if not query:
            print("Please enter a valid query.")
            continue
            
        # Generate and print response
        try:
            result = rag.generate_response(query)
            print(f"\nResponse: {result['response']}")
            print(f"\nPrimary Source: {result['primary_source']}")
            print(f"Retrieval Time: {result['retrieval_time']:.4f} seconds")
            print(f"Generation Time: {result['generation_time']:.4f} seconds")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
