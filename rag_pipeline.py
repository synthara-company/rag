import os
import json
import torch
import faiss
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

class RAGTransformer:
    def __init__(self,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 generator_model: str = 'google/flan-t5-small'):
        """
        Initialize RAG Transformer with embedding and generation models

        Args:
            embedding_model (str): Model for creating document embeddings
            generator_model (str): Model for text generation
        """
        # Embedding Model for Retrieval
        self.embedding_model = SentenceTransformer(embedding_model)

        # Generator Model
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model)

        # Knowledge Base
        self.knowledge_base = []
        self.index = None

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

        # Load sci-fi movies dataset
        sci_fi_movies_path = os.path.join(dataset_dir, 'sci_fi_movies.json')
        cosmos_content_path = os.path.join(dataset_dir, 'cosmos_content.json')

        try:
            with open(sci_fi_movies_path, 'r') as f:
                sci_fi_movies = json.load(f)
                # Extract movie titles and overviews
                movie_documents = [
                    f"Science Fiction Movie: {movie['title']}. Release Date: {movie.get('release_date', 'Unknown')}. Overview: {movie['overview']}. Popularity: {movie.get('popularity', 'N/A')}. Genres: {', '.join(movie.get('genres', []))}"
                    for movie in sci_fi_movies
                ]
                print(f"Loaded {len(movie_documents)} sci-fi movie documents")
        except FileNotFoundError:
            print(f"Sci-fi movies dataset not found at {sci_fi_movies_path}")
            movie_documents = []

        try:
            with open(cosmos_content_path, 'r') as f:
                cosmos_content = json.load(f)
                # Extract cosmos content details
                cosmos_documents = [
                    f"Cosmos Observation Date: {content.get('date', 'Unknown')}. Title: {content.get('title', 'No Title')}. Media Type: {content.get('media_type', 'Unknown')}. Explanation: {content.get('explanation', 'No details')}"
                    for content in cosmos_content
                ]
                print(f"Loaded {len(cosmos_documents)} cosmos content documents")
        except FileNotFoundError:
            print(f"Cosmos content dataset not found at {cosmos_content_path}")
            cosmos_documents = []

        # Combine all documents
        all_documents = ml_documents + movie_documents + cosmos_documents

        # Add documents to knowledge base
        self.add_documents(all_documents)

    def add_documents(self, documents: List[str]):
        """
        Add documents to the knowledge base and create FAISS index

        Args:
            documents (List[str]): List of documents to add
        """
        # Embed documents
        embeddings = self.embedding_model.encode(documents)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # Store documents
        self.knowledge_base.extend(documents)

    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve most relevant documents for a given query

        Args:
            query (str): Input query
            top_k (int): Number of documents to retrieve

        Returns:
            List[str]: Most relevant documents
        """
        # Handle very short or non-informative queries
        if len(query.split()) < 2:
            return self.knowledge_base[:top_k]

        query_embedding = self.embedding_model.encode([query])

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        # Retrieve documents
        retrieved_docs = [self.knowledge_base[idx] for idx in indices[0]]
        return retrieved_docs

    def generate_response(self, query: str) -> str:
        """
        Generate a response using retrieved context

        Args:
            query (str): Input query

        Returns:
            str: Generated response
        """
        # Handle greeting-like queries
        greetings = ['hi', 'hello', 'hey', 'greetings']
        if query.lower().split()[0] in greetings:
            return "Hello! I'm an AI assistant with knowledge about machine learning, sci-fi movies, and cosmos. How can I help you today?"

        # Retrieve context
        context_docs = self.retrieve_context(query)

        # Prepare input
        context = " ".join(context_docs)
        input_text = f"Context: {context}\nQuery: {query}\nProvide a concise and informative response based on the context:"

        # Generate response
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.generator.generate(**inputs,
                                          max_length=150,
                                          num_return_sequences=1,
                                          do_sample=True,
                                          temperature=0.7)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Fallback for very generic queries
        if not response or len(response.split()) < 3:
            response = "I apologize, but I couldn't generate a specific response. Could you please rephrase your query about machine learning, sci-fi movies, or cosmos?"

        return response

def main():
    # Initialize RAG Transformer
    rag = RAGTransformer()

    # Load datasets
    rag.load_datasets()

    # Interactive query loop
    print("RAG Transformer - ML, Sci-Fi, and Cosmos Assistant")
    print("Type 'exit' to quit the program")
    print("Type 'help' for usage instructions")

    while True:
        # Get user input
        query = input("\nEnter your query (or 'exit'/'help' to interact): ").strip()

        # Check for special commands
        if query.lower() == 'exit':
            print("Exiting RAG Transformer. Goodbye!")
            break

        if query.lower() == 'help':
            print("\nThis is an AI Assistant covering:")
            print("- Machine Learning concepts")
            print("- Science Fiction Movies")
            print("- Cosmos and Astronomy")
            print("Ask about AI, movies, space, or scientific topics!")
            continue

        # Check for empty query
        if not query:
            print("Please enter a valid query.")
            continue

        # Generate and print response
        try:
            response = rag.generate_response(query)
            print(f"\nResponse: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
