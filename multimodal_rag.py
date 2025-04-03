"""
Multi-modal RAG system for the RAG Transformer.
This module provides functionality to handle both text and image inputs.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image
import io
import base64
import requests
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MultiModalRAG:
    """Class for multi-modal retrieval-augmented generation"""
    
    def __init__(self,
                 text_embedding_model: str = 'all-MiniLM-L6-v2',
                 vision_model: str = 'openai/clip-vit-base-patch32',
                 generator_model: str = 'google/flan-t5-small'):
        """
        Initialize multi-modal RAG system
        
        Args:
            text_embedding_model (str): Model for text embeddings
            vision_model (str): Model for image embeddings
            generator_model (str): Model for text generation
        """
        # Initialize text embedding model
        try:
            self.text_embedding_model = SentenceTransformer(text_embedding_model)
            print(f"Loaded text embedding model: {text_embedding_model}")
        except Exception as e:
            print(f"Error loading text embedding model: {e}")
            raise
        
        # Initialize vision model
        try:
            self.vision_processor = CLIPProcessor.from_pretrained(vision_model)
            self.vision_model = CLIPModel.from_pretrained(vision_model)
            print(f"Loaded vision model: {vision_model}")
        except Exception as e:
            print(f"Error loading vision model: {e}")
            raise
        
        # Initialize generator model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model)
            print(f"Loaded generator model: {generator_model}")
        except Exception as e:
            print(f"Error loading generator model: {e}")
            raise
        
        # Storage for documents and images
        self.text_documents = []
        self.text_sources = []
        self.text_metadata = []
        self.text_index = None
        
        self.images = []
        self.image_captions = []
        self.image_sources = []
        self.image_metadata = []
        self.image_index = None
    
    def add_text_documents(self,
                           documents: List[str],
                           sources: Optional[List[str]] = None,
                           metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add text documents to the system
        
        Args:
            documents (List[str]): List of text documents
            sources (Optional[List[str]]): List of document sources
            metadata (Optional[List[Dict[str, Any]]]): List of document metadata
        """
        if not documents:
            return
        
        # Store documents
        self.text_documents.extend(documents)
        
        # Store sources
        if sources:
            if len(sources) != len(documents):
                sources = [sources[0] if sources else "Unknown"] * len(documents)
            self.text_sources.extend(sources)
        else:
            self.text_sources.extend(["Unknown"] * len(documents))
        
        # Store metadata
        if metadata:
            if len(metadata) != len(documents):
                metadata = [metadata[0] if metadata else {}] * len(documents)
            self.text_metadata.extend(metadata)
        else:
            self.text_metadata.extend([{}] * len(documents))
        
        # Rebuild text index
        self._build_text_index()
    
    def add_images(self,
                   images: List[Union[str, Image.Image, bytes]],
                   captions: Optional[List[str]] = None,
                   sources: Optional[List[str]] = None,
                   metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add images to the system
        
        Args:
            images (List[Union[str, Image.Image, bytes]]): List of images (file paths, PIL images, or bytes)
            captions (Optional[List[str]]): List of image captions
            sources (Optional[List[str]]): List of image sources
            metadata (Optional[List[Dict[str, Any]]]): List of image metadata
        """
        if not images:
            return
        
        # Process and store images
        processed_images = []
        for img in images:
            if isinstance(img, str):
                # Assume it's a file path or URL
                if img.startswith(('http://', 'https://')):
                    try:
                        response = requests.get(img)
                        response.raise_for_status()
                        img_data = Image.open(io.BytesIO(response.content))
                        processed_images.append(img_data)
                    except Exception as e:
                        print(f"Error loading image from URL {img}: {e}")
                        continue
                else:
                    try:
                        img_data = Image.open(img)
                        processed_images.append(img_data)
                    except Exception as e:
                        print(f"Error loading image from file {img}: {e}")
                        continue
            elif isinstance(img, bytes):
                try:
                    img_data = Image.open(io.BytesIO(img))
                    processed_images.append(img_data)
                except Exception as e:
                    print(f"Error loading image from bytes: {e}")
                    continue
            elif isinstance(img, Image.Image):
                processed_images.append(img)
            else:
                print(f"Unsupported image type: {type(img)}")
                continue
        
        # Store images
        self.images.extend(processed_images)
        
        # Store captions
        if captions:
            if len(captions) != len(processed_images):
                captions = ["No caption"] * len(processed_images)
            self.image_captions.extend(captions)
        else:
            # Generate captions if not provided
            generated_captions = self._generate_image_captions(processed_images)
            self.image_captions.extend(generated_captions)
        
        # Store sources
        if sources:
            if len(sources) != len(processed_images):
                sources = [sources[0] if sources else "Unknown"] * len(processed_images)
            self.image_sources.extend(sources)
        else:
            self.image_sources.extend(["Unknown"] * len(processed_images))
        
        # Store metadata
        if metadata:
            if len(metadata) != len(processed_images):
                metadata = [metadata[0] if metadata else {}] * len(processed_images)
            self.image_metadata.extend(metadata)
        else:
            self.image_metadata.extend([{}] * len(processed_images))
        
        # Rebuild image index
        self._build_image_index()
    
    def _generate_image_captions(self, images: List[Image.Image]) -> List[str]:
        """
        Generate captions for images using CLIP
        
        Args:
            images (List[Image.Image]): List of PIL images
            
        Returns:
            List[str]: Generated captions
        """
        captions = []
        
        # Define candidate labels
        candidate_labels = [
            "a photo of a person",
            "a photo of an animal",
            "a photo of a landscape",
            "a photo of a building",
            "a photo of food",
            "a photo of a vehicle",
            "a photo of technology",
            "a chart or diagram",
            "a screenshot",
            "a painting or artwork"
        ]
        
        for img in images:
            try:
                # Prepare image for CLIP
                inputs = self.vision_processor(
                    text=candidate_labels,
                    images=img,
                    return_tensors="pt",
                    padding=True
                )
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.vision_model(**inputs)
                
                # Get the best label
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                best_label_idx = probs.argmax().item()
                
                caption = candidate_labels[best_label_idx]
                captions.append(caption)
            except Exception as e:
                print(f"Error generating caption: {e}")
                captions.append("No caption")
        
        return captions
    
    def _build_text_index(self) -> None:
        """Build FAISS index for text documents"""
        if not self.text_documents:
            print("No text documents to index")
            return
        
        # Embed documents
        embeddings = self.text_embedding_model.encode(self.text_documents)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.text_index = faiss.IndexFlatL2(dimension)
        self.text_index.add(embeddings)
        
        print(f"Built text index with {len(self.text_documents)} documents")
    
    def _build_image_index(self) -> None:
        """Build FAISS index for images"""
        if not self.images:
            print("No images to index")
            return
        
        # Embed images
        embeddings = []
        batch_size = 8  # Process images in batches
        
        for i in range(0, len(self.images), batch_size):
            batch = self.images[i:i+batch_size]
            
            try:
                # Prepare images for CLIP
                inputs = self.vision_processor(
                    images=batch,
                    return_tensors="pt",
                    padding=True
                )
                
                # Get image embeddings
                with torch.no_grad():
                    outputs = self.vision_model.get_image_features(**inputs)
                
                # Add embeddings to list
                batch_embeddings = outputs.cpu().numpy()
                embeddings.append(batch_embeddings)
            except Exception as e:
                print(f"Error embedding images batch {i}-{i+batch_size}: {e}")
                # Add zero embeddings for failed batch
                zero_embeddings = np.zeros((len(batch), self.vision_model.config.projection_dim))
                embeddings.append(zero_embeddings)
        
        # Combine all embeddings
        all_embeddings = np.vstack(embeddings)
        
        # Create FAISS index
        dimension = all_embeddings.shape[1]
        self.image_index = faiss.IndexFlatL2(dimension)
        self.image_index.add(all_embeddings)
        
        print(f"Built image index with {len(self.images)} images")
    
    def retrieve_text(self,
                      query: str,
                      top_k: int = 5) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
        """
        Retrieve text documents for a query
        
        Args:
            query (str): Query string
            top_k (int): Number of documents to retrieve
            
        Returns:
            Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
                Retrieved documents, their sources, metadata, and scores
        """
        if not self.text_documents or not self.text_index:
            return [], [], [], []
        
        try:
            # Encode query
            query_embedding = self.text_embedding_model.encode([query])
            
            # Search in FAISS index
            distances, indices = self.text_index.search(query_embedding, min(top_k, len(self.text_documents)))
            
            # Convert distances to similarity scores (lower distance = higher similarity)
            max_dist = np.max(distances[0]) if distances[0].size > 0 else 1.0
            similarity_scores = [1.0 - (dist / max_dist) for dist in distances[0]]
            
            # Get documents, sources, and metadata
            retrieved_docs = [self.text_documents[idx] for idx in indices[0]]
            retrieved_sources = [self.text_sources[idx] for idx in indices[0]]
            retrieved_metadata = [self.text_metadata[idx] for idx in indices[0]]
            
            return retrieved_docs, retrieved_sources, retrieved_metadata, similarity_scores
        except Exception as e:
            print(f"Error retrieving text documents: {e}")
            return [], [], [], []
    
    def retrieve_images(self,
                        query: str,
                        top_k: int = 5) -> Tuple[List[Image.Image], List[str], List[str], List[Dict[str, Any]], List[float]]:
        """
        Retrieve images for a text query
        
        Args:
            query (str): Query string
            top_k (int): Number of images to retrieve
            
        Returns:
            Tuple[List[Image.Image], List[str], List[str], List[Dict[str, Any]], List[float]]:
                Retrieved images, their captions, sources, metadata, and scores
        """
        if not self.images or not self.image_index:
            return [], [], [], [], []
        
        try:
            # Encode query
            inputs = self.vision_processor(text=[query], return_tensors="pt", padding=True)
            with torch.no_grad():
                query_embedding = self.vision_model.get_text_features(**inputs).cpu().numpy()
            
            # Search in FAISS index
            distances, indices = self.image_index.search(query_embedding, min(top_k, len(self.images)))
            
            # Convert distances to similarity scores (lower distance = higher similarity)
            max_dist = np.max(distances[0]) if distances[0].size > 0 else 1.0
            similarity_scores = [1.0 - (dist / max_dist) for dist in distances[0]]
            
            # Get images, captions, sources, and metadata
            retrieved_images = [self.images[idx] for idx in indices[0]]
            retrieved_captions = [self.image_captions[idx] for idx in indices[0]]
            retrieved_sources = [self.image_sources[idx] for idx in indices[0]]
            retrieved_metadata = [self.image_metadata[idx] for idx in indices[0]]
            
            return retrieved_images, retrieved_captions, retrieved_sources, retrieved_metadata, similarity_scores
        except Exception as e:
            print(f"Error retrieving images: {e}")
            return [], [], [], [], []
    
    def retrieve_by_image(self,
                          image: Union[str, Image.Image, bytes],
                          top_k: int = 5) -> Tuple[List[Image.Image], List[str], List[str], List[Dict[str, Any]], List[float]]:
        """
        Retrieve similar images for an input image
        
        Args:
            image (Union[str, Image.Image, bytes]): Input image
            top_k (int): Number of images to retrieve
            
        Returns:
            Tuple[List[Image.Image], List[str], List[str], List[Dict[str, Any]], List[float]]:
                Retrieved images, their captions, sources, metadata, and scores
        """
        if not self.images or not self.image_index:
            return [], [], [], [], []
        
        # Process input image
        if isinstance(image, str):
            # Assume it's a file path or URL
            if image.startswith(('http://', 'https://')):
                try:
                    response = requests.get(image)
                    response.raise_for_status()
                    img_data = Image.open(io.BytesIO(response.content))
                except Exception as e:
                    print(f"Error loading image from URL {image}: {e}")
                    return [], [], [], [], []
            else:
                try:
                    img_data = Image.open(image)
                except Exception as e:
                    print(f"Error loading image from file {image}: {e}")
                    return [], [], [], [], []
        elif isinstance(image, bytes):
            try:
                img_data = Image.open(io.BytesIO(image))
            except Exception as e:
                print(f"Error loading image from bytes: {e}")
                return [], [], [], [], []
        elif isinstance(image, Image.Image):
            img_data = image
        else:
            print(f"Unsupported image type: {type(image)}")
            return [], [], [], [], []
        
        try:
            # Encode image
            inputs = self.vision_processor(images=img_data, return_tensors="pt")
            with torch.no_grad():
                image_embedding = self.vision_model.get_image_features(**inputs).cpu().numpy()
            
            # Search in FAISS index
            distances, indices = self.image_index.search(image_embedding, min(top_k, len(self.images)))
            
            # Convert distances to similarity scores (lower distance = higher similarity)
            max_dist = np.max(distances[0]) if distances[0].size > 0 else 1.0
            similarity_scores = [1.0 - (dist / max_dist) for dist in distances[0]]
            
            # Get images, captions, sources, and metadata
            retrieved_images = [self.images[idx] for idx in indices[0]]
            retrieved_captions = [self.image_captions[idx] for idx in indices[0]]
            retrieved_sources = [self.image_sources[idx] for idx in indices[0]]
            retrieved_metadata = [self.image_metadata[idx] for idx in indices[0]]
            
            return retrieved_images, retrieved_captions, retrieved_sources, retrieved_metadata, similarity_scores
        except Exception as e:
            print(f"Error retrieving images by image: {e}")
            return [], [], [], [], []
    
    def generate_response(self,
                          query: str,
                          include_images: bool = True,
                          text_top_k: int = 5,
                          image_top_k: int = 3) -> Dict[str, Any]:
        """
        Generate a response for a query using both text and image retrieval
        
        Args:
            query (str): Query string
            include_images (bool): Whether to include image retrieval
            text_top_k (int): Number of text documents to retrieve
            image_top_k (int): Number of images to retrieve
            
        Returns:
            Dict[str, Any]: Response data including text, images, and metadata
        """
        start_time = time.time()
        
        # Retrieve text documents
        text_docs, text_sources, text_metadata, text_scores = self.retrieve_text(query, text_top_k)
        
        # Retrieve images if requested
        images = []
        image_captions = []
        image_sources = []
        image_metadata = []
        image_scores = []
        
        if include_images and self.images:
            images, image_captions, image_sources, image_metadata, image_scores = self.retrieve_images(query, image_top_k)
        
        # Prepare context for generation
        context = ""
        
        # Add text documents to context
        if text_docs:
            context += "Text information:\n"
            for i, (doc, source) in enumerate(zip(text_docs, text_sources)):
                context += f"[{source}] {doc}\n\n"
        
        # Add image captions to context
        if image_captions:
            context += "Image information:\n"
            for i, (caption, source) in enumerate(zip(image_captions, image_sources)):
                context += f"[{source}] Image {i+1}: {caption}\n"
        
        # Generate response
        gen_start_time = time.time()
        
        input_text = f"""Context: {context}

Query: {query}

Instructions:
1. Provide a concise and informative response based on the context.
2. Focus on directly answering the query.
3. If the context mentions images, refer to them appropriately.
4. If the context doesn't contain relevant information, say so.
5. Do not make up information.

Response:"""

        # Generate response with the model
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
        
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_time = time.time() - gen_start_time
        
        # Convert images to base64 for response
        image_data = []
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_data.append(img_str)
        
        # Prepare response
        response = {
            "text": response_text,
            "images": image_data,
            "image_captions": image_captions,
            "text_sources": text_sources,
            "image_sources": image_sources,
            "text_scores": text_scores,
            "image_scores": image_scores,
            "generation_time": generation_time,
            "total_time": time.time() - start_time
        }
        
        return response


# Example usage
if __name__ == "__main__":
    # Initialize multi-modal RAG
    mm_rag = MultiModalRAG()
    
    # Add text documents
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on the use of data and algorithms to imitate the way that humans learn.",
        "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
        "Transformers are a type of neural network architecture used in NLP that can handle sequential data more effectively than previous models.",
        "Neural networks are computing systems inspired by biological neural networks that can learn to perform tasks by considering examples."
    ]
    
    sources = ["ML"] * len(documents)
    mm_rag.add_text_documents(documents, sources)
    
    # Add images (example with local files)
    # image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
    # mm_rag.add_images(image_paths, sources=["Image Source"])
    
    # Test retrieval and generation
    query = "What is deep learning?"
    response = mm_rag.generate_response(query)
    
    print(f"Query: {query}")
    print(f"Response: {response['text']}")
    print(f"Generation time: {response['generation_time']:.4f} seconds")
    print(f"Total time: {response['total_time']:.4f} seconds")
