"""
Unified RAG System for the RAG Transformer.
This module integrates hybrid retrieval, multi-modal capabilities, personalization, and vector database integration.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image
import io
import base64
import torch
from dotenv import load_dotenv
from datetime import datetime
import uuid

# Import components
from hybrid_retrieval import HybridRetriever
from multimodal_rag import MultiModalRAG
from personalization import PersonalizationEngine, UserProfile
from vector_db_integration import create_vector_db, VectorDBInterface
from advanced_prompting import PromptTemplate, ResponseEvaluator
from model_integrations import ModelFactory, BaseLanguageModel

# Load environment variables
load_dotenv()

class UnifiedRAGSystem:
    """Unified RAG System integrating multiple components"""
    
    def __init__(self,
                 config_path: Optional[str] = None,
                 use_hybrid_retrieval: bool = True,
                 use_multimodal: bool = True,
                 use_personalization: bool = True,
                 use_vector_db: bool = True,
                 vector_db_type: str = 'chroma'):
        """
        Initialize Unified RAG System
        
        Args:
            config_path (Optional[str]): Path to configuration file
            use_hybrid_retrieval (bool): Whether to use hybrid retrieval
            use_multimodal (bool): Whether to use multi-modal capabilities
            use_personalization (bool): Whether to use personalization
            use_vector_db (bool): Whether to use vector database
            vector_db_type (str): Type of vector database ('pinecone', 'chroma', 'qdrant')
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components based on configuration
        self.use_hybrid_retrieval = use_hybrid_retrieval
        self.use_multimodal = use_multimodal
        self.use_personalization = use_personalization
        self.use_vector_db = use_vector_db
        
        # Initialize embedding model
        embedding_model = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        
        # Initialize retriever
        if use_hybrid_retrieval:
            self.retriever = HybridRetriever(
                embedding_model=embedding_model,
                use_bm25=self.config.get('use_bm25', True),
                use_tfidf=self.config.get('use_tfidf', True),
                alpha=self.config.get('retrieval_alpha', 0.7)
            )
            print("Initialized Hybrid Retriever")
        else:
            self.retriever = None
        
        # Initialize multi-modal RAG
        if use_multimodal:
            self.multimodal_rag = MultiModalRAG(
                text_embedding_model=embedding_model,
                vision_model=self.config.get('vision_model', 'openai/clip-vit-base-patch32'),
                generator_model=self.config.get('generator_model', 'google/flan-t5-small')
            )
            print("Initialized Multi-Modal RAG")
        else:
            self.multimodal_rag = None
        
        # Initialize personalization engine
        if use_personalization:
            self.personalization = PersonalizationEngine(
                embedding_model=embedding_model,
                profiles_dir=self.config.get('profiles_dir', 'user_profiles')
            )
            print("Initialized Personalization Engine")
        else:
            self.personalization = None
        
        # Initialize vector database
        if use_vector_db:
            self.vector_db = create_vector_db(
                vector_db_type,
                embedding_model=embedding_model,
                **self.config.get('vector_db_config', {})
            )
            print(f"Initialized Vector Database: {vector_db_type}")
        else:
            self.vector_db = None
        
        # Initialize prompt template and response evaluator
        self.prompt_template = PromptTemplate()
        self.response_evaluator = ResponseEvaluator()
        
        # Initialize model factory
        self.model_factory = ModelFactory()
        
        # Initialize generator model
        generator_model_type = self.config.get('generator_model_type', 'huggingface')
        generator_model_name = self.config.get('generator_model_name', 'google/flan-t5-small')
        
        self.generator = self.model_factory.create_model(
            generator_model_type,
            generator_model_name
        )
        print(f"Initialized Generator Model: {generator_model_type}/{generator_model_name}")
        
        # Storage for documents and metadata
        self.documents = []
        self.document_sources = []
        self.document_metadata = []
        
        # Storage for images
        self.images = []
        self.image_captions = []
        self.image_sources = []
        self.image_metadata = []
        
        # Performance metrics
        self.retrieval_times = []
        self.generation_times = []
        self.evaluation_times = []
        self.total_times = []
        
        # Query history
        self.query_history = []
        self.response_history = []
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Args:
            config_path (Optional[str]): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        default_config = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'vision_model': 'openai/clip-vit-base-patch32',
            'generator_model': 'google/flan-t5-small',
            'generator_model_type': 'huggingface',
            'generator_model_name': 'google/flan-t5-small',
            'use_bm25': True,
            'use_tfidf': True,
            'retrieval_alpha': 0.7,
            'profiles_dir': 'user_profiles',
            'vector_db_config': {
                'persist_directory': 'vector_db'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Merge with default config
                for key, value in config.items():
                    default_config[key] = value
                
                print(f"Loaded configuration from {config_path}")
            except Exception as e:
                print(f"Error loading configuration: {e}")
        
        return default_config
    
    def add_documents(self,
                      documents: List[str],
                      sources: Optional[List[str]] = None,
                      metadata: Optional[List[Dict[str, Any]]] = None,
                      collection_name: str = "default") -> List[str]:
        """
        Add documents to the system
        
        Args:
            documents (List[str]): List of documents
            sources (Optional[List[str]]): List of document sources
            metadata (Optional[List[Dict[str, Any]]]): List of document metadata
            collection_name (str): Name of the collection (for vector database)
            
        Returns:
            List[str]: List of document IDs
        """
        if not documents:
            return []
        
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
        
        # Add documents to hybrid retriever
        if self.use_hybrid_retrieval and self.retriever:
            self.retriever.add_documents(documents, sources, metadata)
        
        # Add documents to multi-modal RAG
        if self.use_multimodal and self.multimodal_rag:
            self.multimodal_rag.add_text_documents(documents, sources, metadata)
        
        # Add documents to vector database
        doc_ids = []
        if self.use_vector_db and self.vector_db:
            doc_ids = self.vector_db.add_documents(documents, metadata, collection_name)
        
        # Store documents locally
        self.documents.extend(documents)
        self.document_sources.extend(sources)
        self.document_metadata.extend(metadata)
        
        return doc_ids
    
    def add_images(self,
                   images: List[Union[str, Image.Image, bytes]],
                   captions: Optional[List[str]] = None,
                   sources: Optional[List[str]] = None,
                   metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add images to the system
        
        Args:
            images (List[Union[str, Image.Image, bytes]]): List of images
            captions (Optional[List[str]]): List of image captions
            sources (Optional[List[str]]): List of image sources
            metadata (Optional[List[Dict[str, Any]]]): List of image metadata
        """
        if not images or not self.use_multimodal or not self.multimodal_rag:
            return
        
        # Add images to multi-modal RAG
        self.multimodal_rag.add_images(images, captions, sources, metadata)
        
        # Store processed images locally (for reference)
        if hasattr(self.multimodal_rag, 'images'):
            self.images = self.multimodal_rag.images
            self.image_captions = self.multimodal_rag.image_captions
            self.image_sources = self.multimodal_rag.image_sources
            self.image_metadata = self.multimodal_rag.image_metadata
    
    def process_query(self,
                      query: str,
                      user_id: Optional[str] = None,
                      include_images: bool = True,
                      text_top_k: int = 5,
                      image_top_k: int = 3,
                      collection_name: str = "default") -> Dict[str, Any]:
        """
        Process a query and generate a response
        
        Args:
            query (str): Query string
            user_id (Optional[str]): User ID for personalization
            include_images (bool): Whether to include images in the response
            text_top_k (int): Number of text documents to retrieve
            image_top_k (int): Number of images to retrieve
            collection_name (str): Name of the collection (for vector database)
            
        Returns:
            Dict[str, Any]: Response data
        """
        start_time = time.time()
        
        # Add query to history
        self.query_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id
        })
        
        # Update user profile if personalization is enabled
        if self.use_personalization and self.personalization and user_id:
            self.personalization.update_profile_from_query(user_id, query)
        
        # Retrieve documents
        retrieval_start = time.time()
        
        retrieved_docs = []
        retrieved_sources = []
        retrieved_metadata = []
        retrieved_scores = []
        retrieval_details = {}
        
        # Choose retrieval method based on configuration
        if self.use_vector_db and self.vector_db:
            # Use vector database for retrieval
            retrieved_docs, retrieved_metadata, retrieved_scores = self.vector_db.search(
                query, text_top_k, collection_name
            )
            
            # Extract sources from metadata
            retrieved_sources = [
                meta.get("source", "Unknown") for meta in retrieved_metadata
            ]
            
            retrieval_details["method"] = "vector_db"
            retrieval_details["db_type"] = self.vector_db.__class__.__name__
        
        elif self.use_hybrid_retrieval and self.retriever:
            # Use hybrid retrieval
            retrieved_docs, retrieved_sources, retrieved_metadata, retrieved_scores, retrieval_details = self.retriever.retrieve(
                query, text_top_k
            )
        
        elif self.use_multimodal and self.multimodal_rag:
            # Use multi-modal RAG for text retrieval
            retrieved_docs, retrieved_sources, retrieved_metadata, retrieved_scores = self.multimodal_rag.retrieve_text(
                query, text_top_k
            )
            
            retrieval_details["method"] = "multimodal_text"
        
        else:
            # Fallback to local documents
            # This is a simple implementation and should be improved for production
            retrieved_docs = self.documents[:text_top_k]
            retrieved_sources = self.document_sources[:text_top_k]
            retrieved_metadata = self.document_metadata[:text_top_k]
            retrieved_scores = [1.0] * len(retrieved_docs)
            
            retrieval_details["method"] = "local_fallback"
        
        retrieval_time = time.time() - retrieval_start
        self.retrieval_times.append(retrieval_time)
        
        # Personalize retrieval results if enabled
        if self.use_personalization and self.personalization and user_id:
            personalization_start = time.time()
            
            retrieved_docs, retrieved_sources, retrieved_scores = self.personalization.personalize_retrieval(
                user_id, retrieved_docs, retrieved_sources, retrieved_scores, text_top_k
            )
            
            retrieval_details["personalized"] = True
            retrieval_details["personalization_time"] = time.time() - personalization_start
        
        # Retrieve images if requested and multi-modal is enabled
        retrieved_images = []
        retrieved_image_captions = []
        retrieved_image_sources = []
        retrieved_image_metadata = []
        retrieved_image_scores = []
        
        if include_images and self.use_multimodal and self.multimodal_rag:
            image_retrieval_start = time.time()
            
            retrieved_images, retrieved_image_captions, retrieved_image_sources, retrieved_image_metadata, retrieved_image_scores = self.multimodal_rag.retrieve_images(
                query, image_top_k
            )
            
            retrieval_details["image_retrieval_time"] = time.time() - image_retrieval_start
        
        # Prepare context for generation
        context = ""
        
        # Add text documents to context
        if retrieved_docs:
            context += "Text information:\n"
            for i, (doc, source, score) in enumerate(zip(retrieved_docs, retrieved_sources, retrieved_scores)):
                context += f"[{source}] (Score: {score:.2f}) {doc}\n\n"
        
        # Add image captions to context
        if retrieved_image_captions:
            context += "Image information:\n"
            for i, (caption, source, score) in enumerate(zip(retrieved_image_captions, retrieved_image_sources, retrieved_image_scores)):
                context += f"[{source}] (Score: {score:.2f}) Image {i+1}: {caption}\n"
        
        # Personalize prompt if enabled
        if self.use_personalization and self.personalization and user_id:
            prompt = self.personalization.personalize_prompt(user_id, context)
        else:
            prompt = context
        
        # Determine primary source for prompt template
        if retrieved_sources:
            source_counter = {}
            for source in retrieved_sources:
                if source in source_counter:
                    source_counter[source] += 1
                else:
                    source_counter[source] = 1
            
            primary_source = max(source_counter.items(), key=lambda x: x[1])[0]
        else:
            primary_source = "General"
        
        # Generate response
        generation_start = time.time()
        
        # Use prompt template to create input for generator
        input_text = self.prompt_template.get_prompt(query, prompt, primary_source)
        
        # Generate response
        generation_result = self.generator.generate(input_text)
        response_text = generation_result.get("text", "")
        
        generation_time = time.time() - generation_start
        self.generation_times.append(generation_time)
        
        # Evaluate and improve response
        evaluation_start = time.time()
        evaluation = self.response_evaluator.evaluate_response(query, context, response_text)
        improved_response = self.response_evaluator.improve_response(query, context, response_text, evaluation)
        evaluation_time = time.time() - evaluation_start
        self.evaluation_times.append(evaluation_time)
        
        # Convert images to base64 for response
        image_data = []
        if retrieved_images:
            for img in retrieved_images:
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                image_data.append(img_str)
        
        # Prepare response
        total_time = time.time() - start_time
        self.total_times.append(total_time)
        
        response = {
            "query": query,
            "response": improved_response,
            "documents": retrieved_docs,
            "sources": retrieved_sources,
            "scores": retrieved_scores,
            "images": image_data,
            "image_captions": retrieved_image_captions,
            "image_sources": retrieved_image_sources,
            "primary_source": primary_source,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "evaluation_time": evaluation_time,
            "total_time": total_time,
            "retrieval_details": retrieval_details,
            "evaluation": evaluation,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add response to history
        self.response_history.append(response)
        
        # Update user profile with response if personalization is enabled
        if self.use_personalization and self.personalization and user_id:
            self.personalization.update_profile_from_query(user_id, query, response)
        
        return response
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the system
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        metrics = {
            "avg_retrieval_time": np.mean(self.retrieval_times) if self.retrieval_times else 0,
            "avg_generation_time": np.mean(self.generation_times) if self.generation_times else 0,
            "avg_evaluation_time": np.mean(self.evaluation_times) if self.evaluation_times else 0,
            "avg_total_time": np.mean(self.total_times) if self.total_times else 0,
            "total_queries": len(self.query_history),
            "document_count": len(self.documents),
            "image_count": len(self.images),
            "components": {
                "hybrid_retrieval": self.use_hybrid_retrieval,
                "multimodal": self.use_multimodal,
                "personalization": self.use_personalization,
                "vector_db": self.use_vector_db
            }
        }
        
        return metrics
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user profile
        
        Args:
            user_id (str): User ID
            
        Returns:
            Optional[Dict[str, Any]]: User profile data
        """
        if not self.use_personalization or not self.personalization:
            return None
        
        profile = self.personalization.get_profile(user_id)
        return profile.to_dict()
    
    def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user statistics
        
        Args:
            user_id (str): User ID
            
        Returns:
            Optional[Dict[str, Any]]: User statistics
        """
        if not self.use_personalization or not self.personalization:
            return None
        
        return self.personalization.get_user_stats(user_id)
    
    def log_user_interaction(self,
                             user_id: str,
                             interaction_type: str,
                             content: str,
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log user interaction
        
        Args:
            user_id (str): User ID
            interaction_type (str): Type of interaction
            content (str): Content of the interaction
            metadata (Optional[Dict[str, Any]]): Additional metadata
        """
        if not self.use_personalization or not self.personalization:
            return
        
        self.personalization.log_interaction(user_id, interaction_type, content, metadata)
    
    def save_state(self, state_path: str) -> bool:
        """
        Save system state to file
        
        Args:
            state_path (str): Path to save state
            
        Returns:
            bool: True if successful
        """
        try:
            state = {
                "config": self.config,
                "query_history": self.query_history,
                "response_history": self.response_history,
                "retrieval_times": self.retrieval_times,
                "generation_times": self.generation_times,
                "evaluation_times": self.evaluation_times,
                "total_times": self.total_times,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            print(f"Saved system state to {state_path}")
            return True
        except Exception as e:
            print(f"Error saving system state: {e}")
            return False
    
    def load_state(self, state_path: str) -> bool:
        """
        Load system state from file
        
        Args:
            state_path (str): Path to load state from
            
        Returns:
            bool: True if successful
        """
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            self.query_history = state.get("query_history", [])
            self.response_history = state.get("response_history", [])
            self.retrieval_times = state.get("retrieval_times", [])
            self.generation_times = state.get("generation_times", [])
            self.evaluation_times = state.get("evaluation_times", [])
            self.total_times = state.get("total_times", [])
            
            print(f"Loaded system state from {state_path}")
            return True
        except Exception as e:
            print(f"Error loading system state: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize unified RAG system
    rag_system = UnifiedRAGSystem(
        use_hybrid_retrieval=True,
        use_multimodal=True,
        use_personalization=True,
        use_vector_db=True,
        vector_db_type='chroma'
    )
    
    # Add documents
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on the use of data and algorithms to imitate the way that humans learn.",
        "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
        "Transformers are a type of neural network architecture used in NLP that can handle sequential data more effectively than previous models.",
        "Neural networks are computing systems inspired by biological neural networks that can learn to perform tasks by considering examples."
    ]
    
    sources = ["ML", "ML", "ML", "ML"]
    
    metadata = [
        {"author": "Author 1", "year": 2020},
        {"author": "Author 2", "year": 2021},
        {"author": "Author 3", "year": 2022},
        {"author": "Author 4", "year": 2023}
    ]
    
    rag_system.add_documents(documents, sources, metadata)
    
    # Process query
    user_id = "user123"
    query = "What is deep learning?"
    
    response = rag_system.process_query(query, user_id)
    
    print(f"Query: {query}")
    print(f"Response: {response['response']}")
    print(f"Retrieval time: {response['retrieval_time']:.4f} seconds")
    print(f"Generation time: {response['generation_time']:.4f} seconds")
    print(f"Total time: {response['total_time']:.4f} seconds")
    
    # Get performance metrics
    metrics = rag_system.get_performance_metrics()
    print(f"\nPerformance metrics: {metrics}")
    
    # Get user profile
    profile = rag_system.get_user_profile(user_id)
    print(f"\nUser profile: {profile}")
    
    # Save system state
    rag_system.save_state("rag_system_state.json")
