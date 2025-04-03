"""
Advanced RAG Transformer with improved prompting, document loading, and response generation.
"""

import os
import json
import torch
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import Counter
import pandas as pd

# Import custom modules
from advanced_prompting import PromptTemplate, ResponseEvaluator
from document_loader import DocumentLoader

# Load environment variables
load_dotenv()

class AdvancedRAGTransformer:
    def __init__(self,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 generator_model: str = 'google/flan-t5-small'):
        """
        Initialize Advanced RAG Transformer with embedding and generation models

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
        self.document_metadata = []  # Track additional metadata
        self.index = None
        
        # Advanced components
        self.prompt_template = PromptTemplate()
        self.response_evaluator = ResponseEvaluator()
        self.document_loader = DocumentLoader()
        
        # Query history
        self.query_history = []
        self.response_history = []
        self.context_history = []
        
        # Performance metrics
        self.retrieval_times = []
        self.generation_times = []
        self.evaluation_times = []

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
            self.document_metadata.append({
                "source_type": "built-in",
                "domain": "Machine Learning"
            })

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
                    self.document_metadata.append({
                        "source_type": "movie",
                        "domain": "Science Fiction",
                        "title": movie['title'],
                        "release_date": movie.get('release_date', 'Unknown'),
                        "genres": movie.get('genres', [])
                    })
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
                    self.document_metadata.append({
                        "source_type": "cosmos",
                        "domain": "Cosmos",
                        "date": content.get('date', 'Unknown'),
                        "title": content.get('title', 'No Title'),
                        "media_type": content.get('media_type', 'Unknown')
                    })
                print(f"Loaded {len(cosmos_content)} cosmos content documents")
        except FileNotFoundError:
            print(f"Cosmos content dataset not found at {cosmos_content_path}")

        # Look for additional datasets
        try:
            for filename in os.listdir(dataset_dir):
                if filename in ['sci_fi_movies.json', 'cosmos_content.json']:
                    continue  # Already loaded
                    
                if filename.endswith('.json'):
                    file_path = os.path.join(dataset_dir, filename)
                    source_name = filename.replace('.json', '').replace('_', ' ').title()
                    
                    try:
                        with open(file_path, 'r') as f:
                            additional_data = json.load(f)
                            
                            if isinstance(additional_data, list):
                                for item in additional_data:
                                    if isinstance(item, dict) and 'content' in item:
                                        self.knowledge_base.append(item['content'])
                                        
                                        # Determine source category
                                        if 'metadata' in item and 'source' in item['metadata']:
                                            source = item['metadata']['source']
                                        else:
                                            source = source_name
                                            
                                        self.document_sources.append(source)
                                        self.document_metadata.append(item.get('metadata', {
                                            "source_type": "additional",
                                            "domain": source
                                        }))
                                
                                print(f"Loaded {len(additional_data)} documents from {filename}")
                    except Exception as e:
                        print(f"Error loading additional dataset {filename}: {e}")
        except Exception as e:
            print(f"Error scanning for additional datasets: {e}")

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

    def add_documents(self, documents: Union[List[str], List[Dict[str, Any]]], source: str = "Custom"):
        """
        Add documents to the knowledge base and update FAISS index

        Args:
            documents (Union[List[str], List[Dict[str, Any]]]): List of documents or document objects
            source (str): Source category of the documents
        """
        if not documents:
            return
            
        # Add documents to knowledge base
        for doc in documents:
            if isinstance(doc, str):
                self.knowledge_base.append(doc)
                self.document_sources.append(source)
                self.document_metadata.append({
                    "source_type": "custom",
                    "domain": source
                })
            elif isinstance(doc, dict) and 'content' in doc:
                self.knowledge_base.append(doc['content'])
                
                # Use metadata if available
                if 'metadata' in doc:
                    metadata = doc['metadata']
                    source_from_metadata = metadata.get('source', source)
                    self.document_sources.append(source_from_metadata)
                    self.document_metadata.append(metadata)
                else:
                    self.document_sources.append(source)
                    self.document_metadata.append({
                        "source_type": "custom",
                        "domain": source
                    })
            
        # Recreate index
        self._create_index()
        
    def add_web_content(self, url: str, source_name: str = None):
        """
        Add content from a web page to the knowledge base
        
        Args:
            url (str): URL to fetch
            source_name (str): Optional name for the source
        
        Returns:
            int: Number of documents added
        """
        documents = self.document_loader.process_web_page(url)
        
        if not documents:
            print(f"No content extracted from {url}")
            return 0
        
        # Use domain as source name if not provided
        if not source_name:
            from urllib.parse import urlparse
            source_name = f"Web: {urlparse(url).netloc}"
        
        self.add_documents(documents, source_name)
        return len(documents)
    
    def add_wikipedia_content(self, topic: str, max_results: int = 3):
        """
        Add Wikipedia content on a topic to the knowledge base
        
        Args:
            topic (str): Topic to search for
            max_results (int): Maximum number of articles to add
        
        Returns:
            int: Number of documents added
        """
        documents = self.document_loader.process_wikipedia_topic(topic, max_results)
        
        if not documents:
            print(f"No Wikipedia content found for {topic}")
            return 0
        
        self.add_documents(documents, f"Wikipedia: {topic}")
        return len(documents)

    def retrieve_context(self, query: str, top_k: int = 5) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
        """
        Retrieve most relevant documents for a given query

        Args:
            query (str): Input query
            top_k (int): Number of documents to retrieve

        Returns:
            Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]: 
                Retrieved documents, their sources, metadata, and relevance scores
        """
        start_time = time.time()
        
        # Handle very short or non-informative queries
        if len(query.split()) < 2:
            docs = self.knowledge_base[:min(top_k, len(self.knowledge_base))]
            sources = self.document_sources[:min(top_k, len(self.document_sources))]
            metadata = self.document_metadata[:min(top_k, len(self.document_metadata))]
            scores = [1.0] * len(docs)  # Default score
            
            self.retrieval_times.append(time.time() - start_time)
            return docs, sources, metadata, scores

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
        retrieved_metadata = [self.document_metadata[idx] for idx in indices[0]]
        
        self.retrieval_times.append(time.time() - start_time)
        return retrieved_docs, retrieved_sources, retrieved_metadata, similarity_scores

    def generate_response(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate a response using retrieved context with advanced prompting

        Args:
            query (str): Input query
            top_k (int): Number of documents to retrieve

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
                self.context_history.append([])
                return {
                    "response": response,
                    "sources": ["System"],
                    "primary_source": "System",
                    "retrieval_time": 0,
                    "generation_time": 0,
                    "evaluation_time": 0,
                    "context_docs": [],
                    "metadata": []
                }

            # Simple question handling
            if "how are you" in query.lower():
                response = "I'm functioning well, thank you for asking! I'm here to help with information about machine learning, sci-fi movies, and cosmos."
                self.response_history.append(response)
                self.context_history.append([])
                return {
                    "response": response,
                    "sources": ["System"],
                    "primary_source": "System",
                    "retrieval_time": 0,
                    "generation_time": 0,
                    "evaluation_time": 0,
                    "context_docs": [],
                    "metadata": []
                }

            # Retrieve context
            context_docs, sources, metadata, scores = self.retrieve_context(query, top_k)
            self.context_history.append(context_docs)
            
            # Determine primary source category
            source_counter = Counter(sources)
            primary_source = source_counter.most_common(1)[0][0] if source_counter else "Unknown"
            
            # Start generation timer
            gen_start_time = time.time()
            
            # Prepare context
            context = " ".join(context_docs)
            
            # Use advanced prompting
            input_text = self.prompt_template.get_prompt(query, context, primary_source)

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
            
            # Evaluate and improve response
            eval_start_time = time.time()
            evaluation = self.response_evaluator.evaluate_response(query, context, response)
            improved_response = self.response_evaluator.improve_response(query, context, response, evaluation)
            evaluation_time = time.time() - eval_start_time
            self.evaluation_times.append(evaluation_time)
            
            # Get the most recent retrieval time
            retrieval_time = self.retrieval_times[-1] if self.retrieval_times else 0

            # Fallback for very generic queries
            if not improved_response or len(improved_response.split()) < 3:
                improved_response = "I apologize, but I couldn't generate a specific response. Could you please rephrase your query about machine learning, sci-fi movies, or cosmos?"
            
            # Add response to history
            self.response_history.append(improved_response)
            
            return {
                "response": improved_response,
                "sources": sources,
                "metadata": metadata,
                "primary_source": primary_source,
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "evaluation_time": evaluation_time,
                "context_docs": context_docs,
                "similarity_scores": scores,
                "evaluation": evaluation
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            response = "I encountered an error while processing your query. Please try a different question about machine learning, sci-fi movies, or cosmos."
            self.response_history.append(response)
            self.context_history.append([])
            return {
                "response": response,
                "sources": ["Error"],
                "primary_source": "Error",
                "retrieval_time": 0,
                "generation_time": 0,
                "evaluation_time": 0,
                "context_docs": [],
                "metadata": []
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
            "avg_evaluation_time": np.mean(self.evaluation_times) if self.evaluation_times else 0,
            "total_queries": len(self.query_history),
            "knowledge_base_size": len(self.knowledge_base),
            "source_distribution": Counter(self.document_sources),
            "domain_distribution": Counter([meta.get("domain", "Unknown") for meta in self.document_metadata])
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
        
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(sources)))
        ax.bar(sources, counts, color=colors)
        ax.set_title('Knowledge Base Source Distribution')
        ax.set_xlabel('Source')
        ax.set_ylabel('Number of Documents')
        
        # Rotate x-axis labels if there are many sources
        if len(sources) > 5:
            plt.xticks(rotation=45, ha='right')
            
        for i, count in enumerate(counts):
            ax.text(i, count + 0.1, str(count), ha='center')
            
        plt.tight_layout()
        return fig
    
    def get_conversation_history(self) -> pd.DataFrame:
        """
        Get conversation history as a DataFrame

        Returns:
            pd.DataFrame: Conversation history
        """
        history = []
        
        for i, (query, response) in enumerate(zip(self.query_history, self.response_history)):
            context = self.context_history[i] if i < len(self.context_history) else []
            
            history.append({
                "turn": i + 1,
                "query": query,
                "response": response,
                "context_count": len(context)
            })
        
        return pd.DataFrame(history)


def main():
    # Initialize Advanced RAG Transformer
    rag = AdvancedRAGTransformer()
    
    # Load datasets
    rag.load_datasets()
    
    # Interactive query loop
    print("Advanced RAG Transformer - ML, Sci-Fi, and Cosmos Assistant")
    print("Type 'exit' to quit the program")
    print("Type 'help' for usage instructions")
    print("Type 'stats' to see performance metrics")
    print("Type 'web <url>' to add web content")
    print("Type 'wiki <topic>' to add Wikipedia content")

    while True:
        # Get user input
        query = input("\nEnter your query: ").strip()
        
        # Check for special commands
        if query.lower() == 'exit':
            print("Exiting Advanced RAG Transformer. Goodbye!")
            break
            
        if query.lower() == 'help':
            print("\nThis is an Advanced AI Assistant covering:")
            print("- Machine Learning concepts")
            print("- Science Fiction Movies")
            print("- Cosmos and Astronomy")
            print("\nSpecial commands:")
            print("- 'stats': View system performance metrics")
            print("- 'web <url>': Add content from a web page")
            print("- 'wiki <topic>': Add Wikipedia content on a topic")
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
            print(f"Average Evaluation Time: {metrics['avg_evaluation_time']:.4f} seconds")
            print("\nSource Distribution:")
            for source, count in metrics['source_distribution'].most_common():
                print(f"- {source}: {count} documents")
            continue
            
        if query.lower().startswith('web '):
            url = query[4:].strip()
            if url:
                print(f"Adding content from {url}...")
                count = rag.add_web_content(url)
                print(f"Added {count} document chunks from {url}")
            else:
                print("Please provide a URL")
            continue
            
        if query.lower().startswith('wiki '):
            topic = query[5:].strip()
            if topic:
                print(f"Adding Wikipedia content for '{topic}'...")
                count = rag.add_wikipedia_content(topic)
                print(f"Added {count} Wikipedia articles about {topic}")
            else:
                print("Please provide a topic")
            continue
            
        # Check for empty query
        if not query:
            print("Please enter a valid query.")
            continue
            
        # Generate and print response
        try:
            print("Generating response...")
            result = rag.generate_response(query)
            
            print(f"\nResponse: {result['response']}")
            print(f"\nPrimary Source: {result['primary_source']}")
            print(f"Retrieval Time: {result['retrieval_time']:.4f} seconds")
            print(f"Generation Time: {result['generation_time']:.4f} seconds")
            print(f"Evaluation Time: {result['evaluation_time']:.4f} seconds")
            
            # Show evaluation if available
            if 'evaluation' in result:
                eval_result = result['evaluation']
                print(f"\nResponse Quality: {eval_result.get('quality', 'Unknown')}")
                if 'issue' in eval_result:
                    print(f"Issue: {eval_result['issue']}")
                if 'suggestion' in eval_result:
                    print(f"Suggestion: {eval_result['suggestion']}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
