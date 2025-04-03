"""
Simple RAG System for demonstration purposes.
This is a simplified version that doesn't depend on all components.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss

class SimpleRAGSystem:
    """Simple RAG System for demonstration purposes"""
    
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 generator_model: str = 'google/flan-t5-small'):
        """
        Initialize Simple RAG System
        
        Args:
            embedding_model (str): Model for creating embeddings
            generator_model (str): Model for text generation
        """
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            print(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise
        
        # Initialize generator model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model)
            print(f"Loaded generator model: {generator_model}")
        except Exception as e:
            print(f"Error loading generator model: {e}")
            raise
        
        # Storage for documents and metadata
        self.documents = []
        self.document_sources = []
        self.document_metadata = []
        self.index = None
        
        # Performance metrics
        self.retrieval_times = []
        self.generation_times = []
        self.total_times = []
        
        # Query history
        self.query_history = []
        self.response_history = []
    
    def add_documents(self,
                      documents: List[str],
                      sources: Optional[List[str]] = None,
                      metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add documents to the system
        
        Args:
            documents (List[str]): List of documents
            sources (Optional[List[str]]): List of document sources
            metadata (Optional[List[Dict[str, Any]]]): List of document metadata
        """
        if not documents:
            return
        
        # Create sources if not provided
        if sources is None:
            sources = ["Unknown"] * len(documents)
        elif len(sources) != len(documents):
            sources = [sources[0] if sources else "Unknown"] * len(documents)
        
        # Create metadata if not provided
        if metadata is None:
            metadata = [{}] * len(documents)
        elif len(metadata) != len(documents):
            metadata = [metadata[0] if metadata else {}] * len(documents)
        
        # Store documents
        self.documents.extend(documents)
        self.document_sources.extend(sources)
        self.document_metadata.extend(metadata)
        
        # Build index
        self._build_index()
    
    def _build_index(self) -> None:
        """Build FAISS index for documents"""
        if not self.documents:
            print("No documents to index")
            return
        
        # Embed documents
        embeddings = self.embedding_model.encode(self.documents)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        print(f"Built index with {len(self.documents)} documents")
    
    def retrieve(self,
                query: str,
                top_k: int = 5) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
        """
        Retrieve documents for a query
        
        Args:
            query (str): Query string
            top_k (int): Number of documents to retrieve
            
        Returns:
            Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
                Retrieved documents, their sources, metadata, and scores
        """
        if not self.documents or not self.index:
            return [], [], [], []
        
        start_time = time.time()
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Convert distances to similarity scores (lower distance = higher similarity)
        max_dist = np.max(distances[0]) if distances[0].size > 0 else 1.0
        similarity_scores = [1.0 - (dist / max_dist) for dist in distances[0]]
        
        # Get documents, sources, and metadata
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        retrieved_sources = [self.document_sources[idx] for idx in indices[0]]
        retrieved_metadata = [self.document_metadata[idx] for idx in indices[0]]
        
        self.retrieval_times.append(time.time() - start_time)
        
        return retrieved_docs, retrieved_sources, retrieved_metadata, similarity_scores
    
    def generate(self, query: str, context: str) -> Tuple[str, float]:
        """
        Generate a response for a query and context
        
        Args:
            query (str): Query string
            context (str): Context string
            
        Returns:
            Tuple[str, float]: Generated response and generation time
        """
        start_time = time.time()
        
        # Prepare input
        input_text = f"""Context: {context}

Query: {query}

Answer:"""
        
        # Generate response
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_length=150,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=3
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        self.generation_times.append(generation_time)
        
        return response, generation_time
    
    def process_query(self,
                      query: str,
                      top_k: int = 5) -> Dict[str, Any]:
        """
        Process a query and generate a response
        
        Args:
            query (str): Query string
            top_k (int): Number of documents to retrieve
            
        Returns:
            Dict[str, Any]: Response data
        """
        start_time = time.time()
        
        # Add query to history
        self.query_history.append(query)
        
        # Retrieve documents
        retrieved_docs, retrieved_sources, retrieved_metadata, retrieved_scores = self.retrieve(query, top_k)
        
        # Prepare context
        context = ""
        if retrieved_docs:
            for i, (doc, source, score) in enumerate(zip(retrieved_docs, retrieved_sources, retrieved_scores)):
                context += f"[{source}] (Score: {score:.2f}) {doc}\n\n"
        
        # Generate response
        response, generation_time = self.generate(query, context)
        
        # Determine primary source
        primary_source = "Unknown"
        if retrieved_sources:
            source_counter = {}
            for source in retrieved_sources:
                if source in source_counter:
                    source_counter[source] += 1
                else:
                    source_counter[source] = 1
            
            primary_source = max(source_counter.items(), key=lambda x: x[1])[0]
        
        # Prepare response data
        total_time = time.time() - start_time
        self.total_times.append(total_time)
        
        result = {
            "query": query,
            "response": response,
            "documents": retrieved_docs,
            "sources": retrieved_sources,
            "metadata": retrieved_metadata,
            "scores": retrieved_scores,
            "primary_source": primary_source,
            "retrieval_time": self.retrieval_times[-1] if self.retrieval_times else 0,
            "generation_time": generation_time,
            "total_time": total_time
        }
        
        # Add response to history
        self.response_history.append(result)
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the system
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        metrics = {
            "avg_retrieval_time": np.mean(self.retrieval_times) if self.retrieval_times else 0,
            "avg_generation_time": np.mean(self.generation_times) if self.generation_times else 0,
            "avg_total_time": np.mean(self.total_times) if self.total_times else 0,
            "total_queries": len(self.query_history),
            "document_count": len(self.documents)
        }
        
        return metrics
    
    def load_sample_documents(self, file_path: str) -> int:
        """
        Load sample documents from a JSON file
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            int: Number of documents loaded
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            documents = data.get('documents', [])
            sources = data.get('sources', [])
            metadata = data.get('metadata', [])
            
            if documents:
                self.add_documents(documents, sources, metadata)
                print(f"Loaded {len(documents)} sample documents")
                return len(documents)
            else:
                print("No documents found in the file")
                return 0
        except Exception as e:
            print(f"Error loading sample documents: {e}")
            return 0


# Example usage
if __name__ == "__main__":
    # Initialize Simple RAG System
    rag_system = SimpleRAGSystem()
    
    # Load sample documents
    rag_system.load_sample_documents("sample_documents.json")
    
    # Process a query
    query = "What is deep learning?"
    result = rag_system.process_query(query)
    
    print(f"Query: {query}")
    print(f"Response: {result['response']}")
    print(f"Primary Source: {result['primary_source']}")
    print(f"Retrieval Time: {result['retrieval_time']:.4f} seconds")
    print(f"Generation Time: {result['generation_time']:.4f} seconds")
    print(f"Total Time: {result['total_time']:.4f} seconds")
