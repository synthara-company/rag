"""
Vector database integration for the RAG Transformer system.
This module provides functionality to store and retrieve embeddings using vector databases.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VectorDBInterface:
    """Abstract base class for vector database integrations"""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize vector database interface
        
        Args:
            embedding_model (str): Model for creating embeddings
        """
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            print(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise
    
    def add_documents(self,
                      documents: List[str],
                      metadata: Optional[List[Dict[str, Any]]] = None,
                      collection_name: str = "default") -> List[str]:
        """
        Add documents to the vector database
        
        Args:
            documents (List[str]): List of documents
            metadata (Optional[List[Dict[str, Any]]]): List of document metadata
            collection_name (str): Name of the collection
            
        Returns:
            List[str]: List of document IDs
        """
        raise NotImplementedError("Subclasses must implement add_documents")
    
    def search(self,
               query: str,
               top_k: int = 5,
               collection_name: str = "default") -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Search for documents in the vector database
        
        Args:
            query (str): Query string
            top_k (int): Number of documents to retrieve
            collection_name (str): Name of the collection
            
        Returns:
            Tuple[List[str], List[Dict[str, Any]], List[float]]:
                Retrieved documents, their metadata, and scores
        """
        raise NotImplementedError("Subclasses must implement search")
    
    def delete_documents(self,
                         document_ids: List[str],
                         collection_name: str = "default") -> bool:
        """
        Delete documents from the vector database
        
        Args:
            document_ids (List[str]): List of document IDs
            collection_name (str): Name of the collection
            
        Returns:
            bool: True if successful
        """
        raise NotImplementedError("Subclasses must implement delete_documents")
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the vector database
        
        Returns:
            List[str]: List of collection names
        """
        raise NotImplementedError("Subclasses must implement list_collections")
    
    def create_collection(self, collection_name: str) -> bool:
        """
        Create a new collection in the vector database
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            bool: True if successful
        """
        raise NotImplementedError("Subclasses must implement create_collection")
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from the vector database
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            bool: True if successful
        """
        raise NotImplementedError("Subclasses must implement delete_collection")


class PineconeDB(VectorDBInterface):
    """Integration with Pinecone vector database"""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 environment: Optional[str] = None,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize Pinecone integration
        
        Args:
            api_key (Optional[str]): Pinecone API key
            environment (Optional[str]): Pinecone environment
            embedding_model (str): Model for creating embeddings
        """
        super().__init__(embedding_model)
        
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT")
        
        if not self.api_key:
            raise ValueError("Pinecone API key not provided")
        
        if not self.environment:
            raise ValueError("Pinecone environment not provided")
        
        try:
            import pinecone
            pinecone.init(api_key=self.api_key, environment=self.environment)
            self.pinecone = pinecone
            print("Connected to Pinecone")
        except ImportError:
            print("Error: pinecone-client package not installed. Install with 'pip install pinecone-client'")
            raise
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}")
            raise
    
    def add_documents(self,
                      documents: List[str],
                      metadata: Optional[List[Dict[str, Any]]] = None,
                      collection_name: str = "default") -> List[str]:
        """
        Add documents to Pinecone
        
        Args:
            documents (List[str]): List of documents
            metadata (Optional[List[Dict[str, Any]]]): List of document metadata
            collection_name (str): Name of the index
            
        Returns:
            List[str]: List of document IDs
        """
        if not documents:
            return []
        
        # Create index if it doesn't exist
        if collection_name not in self.pinecone.list_indexes():
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.pinecone.create_index(
                name=collection_name,
                dimension=dimension,
                metric="cosine"
            )
            print(f"Created Pinecone index: {collection_name}")
        
        # Get index
        index = self.pinecone.Index(collection_name)
        
        # Generate document IDs
        doc_ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Create metadata if not provided
        if metadata is None:
            metadata = [{"text": doc} for doc in documents]
        elif len(metadata) != len(documents):
            metadata = [{"text": doc} for doc in documents]
        
        # Add text to metadata
        for i, doc in enumerate(documents):
            metadata[i]["text"] = doc
        
        # Embed documents
        embeddings = self.embedding_model.encode(documents)
        
        # Prepare vectors
        vectors = []
        for i, (doc_id, embedding, meta) in enumerate(zip(doc_ids, embeddings, metadata)):
            vectors.append({
                "id": doc_id,
                "values": embedding.tolist(),
                "metadata": meta
            })
        
        # Upsert vectors in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)
        
        print(f"Added {len(documents)} documents to Pinecone index: {collection_name}")
        return doc_ids
    
    def search(self,
               query: str,
               top_k: int = 5,
               collection_name: str = "default") -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Search for documents in Pinecone
        
        Args:
            query (str): Query string
            top_k (int): Number of documents to retrieve
            collection_name (str): Name of the index
            
        Returns:
            Tuple[List[str], List[Dict[str, Any]], List[float]]:
                Retrieved documents, their metadata, and scores
        """
        # Check if index exists
        if collection_name not in self.pinecone.list_indexes():
            print(f"Pinecone index not found: {collection_name}")
            return [], [], []
        
        # Get index
        index = self.pinecone.Index(collection_name)
        
        # Embed query
        query_embedding = self.embedding_model.encode(query)
        
        # Search in Pinecone
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract documents, metadata, and scores
        documents = []
        metadata_list = []
        scores = []
        
        for match in results["matches"]:
            doc_id = match["id"]
            score = match["score"]
            meta = match["metadata"]
            
            # Extract text from metadata
            text = meta.get("text", "")
            
            # Remove text from metadata to avoid duplication
            meta_copy = meta.copy()
            if "text" in meta_copy:
                del meta_copy["text"]
            
            documents.append(text)
            metadata_list.append(meta_copy)
            scores.append(score)
        
        return documents, metadata_list, scores
    
    def delete_documents(self,
                         document_ids: List[str],
                         collection_name: str = "default") -> bool:
        """
        Delete documents from Pinecone
        
        Args:
            document_ids (List[str]): List of document IDs
            collection_name (str): Name of the index
            
        Returns:
            bool: True if successful
        """
        # Check if index exists
        if collection_name not in self.pinecone.list_indexes():
            print(f"Pinecone index not found: {collection_name}")
            return False
        
        # Get index
        index = self.pinecone.Index(collection_name)
        
        # Delete vectors
        index.delete(ids=document_ids)
        
        print(f"Deleted {len(document_ids)} documents from Pinecone index: {collection_name}")
        return True
    
    def list_collections(self) -> List[str]:
        """
        List all indexes in Pinecone
        
        Returns:
            List[str]: List of index names
        """
        return self.pinecone.list_indexes()
    
    def create_collection(self, collection_name: str) -> bool:
        """
        Create a new index in Pinecone
        
        Args:
            collection_name (str): Name of the index
            
        Returns:
            bool: True if successful
        """
        # Check if index already exists
        if collection_name in self.pinecone.list_indexes():
            print(f"Pinecone index already exists: {collection_name}")
            return False
        
        # Create index
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.pinecone.create_index(
            name=collection_name,
            dimension=dimension,
            metric="cosine"
        )
        
        print(f"Created Pinecone index: {collection_name}")
        return True
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete an index from Pinecone
        
        Args:
            collection_name (str): Name of the index
            
        Returns:
            bool: True if successful
        """
        # Check if index exists
        if collection_name not in self.pinecone.list_indexes():
            print(f"Pinecone index not found: {collection_name}")
            return False
        
        # Delete index
        self.pinecone.delete_index(collection_name)
        
        print(f"Deleted Pinecone index: {collection_name}")
        return True


class ChromaDB(VectorDBInterface):
    """Integration with ChromaDB vector database"""
    
    def __init__(self, 
                 persist_directory: str = "chroma_db",
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize ChromaDB integration
        
        Args:
            persist_directory (str): Directory to persist ChromaDB
            embedding_model (str): Model for creating embeddings
        """
        super().__init__(embedding_model)
        
        self.persist_directory = persist_directory
        
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=persist_directory)
            print(f"Connected to ChromaDB at {persist_directory}")
        except ImportError:
            print("Error: chromadb package not installed. Install with 'pip install chromadb'")
            raise
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            raise
    
    def add_documents(self,
                      documents: List[str],
                      metadata: Optional[List[Dict[str, Any]]] = None,
                      collection_name: str = "default") -> List[str]:
        """
        Add documents to ChromaDB
        
        Args:
            documents (List[str]): List of documents
            metadata (Optional[List[Dict[str, Any]]]): List of document metadata
            collection_name (str): Name of the collection
            
        Returns:
            List[str]: List of document IDs
        """
        if not documents:
            return []
        
        # Get or create collection
        try:
            collection = self.client.get_or_create_collection(collection_name)
        except Exception as e:
            print(f"Error getting or creating collection: {e}")
            return []
        
        # Generate document IDs
        doc_ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Create metadata if not provided
        if metadata is None:
            metadata = [{} for _ in range(len(documents))]
        elif len(metadata) != len(documents):
            metadata = [{} for _ in range(len(documents))]
        
        # Embed documents
        embeddings = self.embedding_model.encode(documents)
        
        # Add documents to collection
        try:
            collection.add(
                ids=doc_ids,
                documents=documents,
                embeddings=[embedding.tolist() for embedding in embeddings],
                metadatas=metadata
            )
            
            print(f"Added {len(documents)} documents to ChromaDB collection: {collection_name}")
            return doc_ids
        except Exception as e:
            print(f"Error adding documents to collection: {e}")
            return []
    
    def search(self,
               query: str,
               top_k: int = 5,
               collection_name: str = "default") -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Search for documents in ChromaDB
        
        Args:
            query (str): Query string
            top_k (int): Number of documents to retrieve
            collection_name (str): Name of the collection
            
        Returns:
            Tuple[List[str], List[Dict[str, Any]], List[float]]:
                Retrieved documents, their metadata, and scores
        """
        # Get collection
        try:
            collection = self.client.get_collection(collection_name)
        except Exception as e:
            print(f"Error getting collection: {e}")
            return [], [], []
        
        # Embed query
        query_embedding = self.embedding_model.encode(query)
        
        # Search in ChromaDB
        try:
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Extract documents, metadata, and scores
            documents = results["documents"][0]
            metadata_list = results["metadatas"][0]
            
            # Convert distances to similarity scores (ChromaDB returns L2 distances)
            distances = results["distances"][0]
            max_dist = max(distances) if distances else 1.0
            scores = [1.0 - (dist / max_dist) for dist in distances]
            
            return documents, metadata_list, scores
        except Exception as e:
            print(f"Error searching in collection: {e}")
            return [], [], []
    
    def delete_documents(self,
                         document_ids: List[str],
                         collection_name: str = "default") -> bool:
        """
        Delete documents from ChromaDB
        
        Args:
            document_ids (List[str]): List of document IDs
            collection_name (str): Name of the collection
            
        Returns:
            bool: True if successful
        """
        # Get collection
        try:
            collection = self.client.get_collection(collection_name)
        except Exception as e:
            print(f"Error getting collection: {e}")
            return False
        
        # Delete documents
        try:
            collection.delete(ids=document_ids)
            print(f"Deleted {len(document_ids)} documents from ChromaDB collection: {collection_name}")
            return True
        except Exception as e:
            print(f"Error deleting documents from collection: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        List all collections in ChromaDB
        
        Returns:
            List[str]: List of collection names
        """
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
    
    def create_collection(self, collection_name: str) -> bool:
        """
        Create a new collection in ChromaDB
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            bool: True if successful
        """
        try:
            self.client.create_collection(collection_name)
            print(f"Created ChromaDB collection: {collection_name}")
            return True
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from ChromaDB
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            bool: True if successful
        """
        try:
            self.client.delete_collection(collection_name)
            print(f"Deleted ChromaDB collection: {collection_name}")
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False


class QdrantDB(VectorDBInterface):
    """Integration with Qdrant vector database"""
    
    def __init__(self, 
                 url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 local_path: Optional[str] = "qdrant_db",
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize Qdrant integration
        
        Args:
            url (Optional[str]): Qdrant server URL
            api_key (Optional[str]): Qdrant API key
            local_path (Optional[str]): Path for local Qdrant storage
            embedding_model (str): Model for creating embeddings
        """
        super().__init__(embedding_model)
        
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.local_path = local_path
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            self.QdrantClient = QdrantClient
            self.Distance = Distance
            self.VectorParams = VectorParams
            
            # Connect to Qdrant
            if self.url:
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
                print(f"Connected to Qdrant at {self.url}")
            else:
                self.client = QdrantClient(path=self.local_path)
                print(f"Connected to local Qdrant at {self.local_path}")
        except ImportError:
            print("Error: qdrant-client package not installed. Install with 'pip install qdrant-client'")
            raise
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            raise
    
    def add_documents(self,
                      documents: List[str],
                      metadata: Optional[List[Dict[str, Any]]] = None,
                      collection_name: str = "default") -> List[str]:
        """
        Add documents to Qdrant
        
        Args:
            documents (List[str]): List of documents
            metadata (Optional[List[Dict[str, Any]]]): List of document metadata
            collection_name (str): Name of the collection
            
        Returns:
            List[str]: List of document IDs
        """
        if not documents:
            return []
        
        # Create collection if it doesn't exist
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=self.VectorParams(size=dimension, distance=self.Distance.COSINE)
            )
            print(f"Created Qdrant collection: {collection_name}")
        
        # Generate document IDs
        doc_ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Create metadata if not provided
        if metadata is None:
            metadata = [{"text": doc} for doc in documents]
        elif len(metadata) != len(documents):
            metadata = [{"text": doc} for doc in documents]
        
        # Add text to metadata
        for i, doc in enumerate(documents):
            metadata[i]["text"] = doc
        
        # Embed documents
        embeddings = self.embedding_model.encode(documents)
        
        # Add documents to collection
        from qdrant_client.models import PointStruct
        
        points = [
            PointStruct(
                id=doc_id,
                vector=embedding.tolist(),
                payload=meta
            )
            for doc_id, embedding, meta in zip(doc_ids, embeddings, metadata)
        ]
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        print(f"Added {len(documents)} documents to Qdrant collection: {collection_name}")
        return doc_ids
    
    def search(self,
               query: str,
               top_k: int = 5,
               collection_name: str = "default") -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Search for documents in Qdrant
        
        Args:
            query (str): Query string
            top_k (int): Number of documents to retrieve
            collection_name (str): Name of the collection
            
        Returns:
            Tuple[List[str], List[Dict[str, Any]], List[float]]:
                Retrieved documents, their metadata, and scores
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            print(f"Qdrant collection not found: {collection_name}")
            return [], [], []
        
        # Embed query
        query_embedding = self.embedding_model.encode(query)
        
        # Search in Qdrant
        from qdrant_client.models import Filter
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        # Extract documents, metadata, and scores
        documents = []
        metadata_list = []
        scores = []
        
        for result in results:
            score = result.score
            payload = result.payload
            
            # Extract text from payload
            text = payload.get("text", "")
            
            # Remove text from payload to avoid duplication
            payload_copy = payload.copy()
            if "text" in payload_copy:
                del payload_copy["text"]
            
            documents.append(text)
            metadata_list.append(payload_copy)
            scores.append(score)
        
        return documents, metadata_list, scores
    
    def delete_documents(self,
                         document_ids: List[str],
                         collection_name: str = "default") -> bool:
        """
        Delete documents from Qdrant
        
        Args:
            document_ids (List[str]): List of document IDs
            collection_name (str): Name of the collection
            
        Returns:
            bool: True if successful
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            print(f"Qdrant collection not found: {collection_name}")
            return False
        
        # Delete points
        self.client.delete(
            collection_name=collection_name,
            points_selector=document_ids
        )
        
        print(f"Deleted {len(document_ids)} documents from Qdrant collection: {collection_name}")
        return True
    
    def list_collections(self) -> List[str]:
        """
        List all collections in Qdrant
        
        Returns:
            List[str]: List of collection names
        """
        collections = self.client.get_collections().collections
        return [collection.name for collection in collections]
    
    def create_collection(self, collection_name: str) -> bool:
        """
        Create a new collection in Qdrant
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            bool: True if successful
        """
        # Check if collection already exists
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name in collection_names:
            print(f"Qdrant collection already exists: {collection_name}")
            return False
        
        # Create collection
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=self.VectorParams(size=dimension, distance=self.Distance.COSINE)
        )
        
        print(f"Created Qdrant collection: {collection_name}")
        return True
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from Qdrant
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            bool: True if successful
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            print(f"Qdrant collection not found: {collection_name}")
            return False
        
        # Delete collection
        self.client.delete_collection(collection_name=collection_name)
        
        print(f"Deleted Qdrant collection: {collection_name}")
        return True


# Factory function to create vector database instances
def create_vector_db(db_type: str, **kwargs) -> VectorDBInterface:
    """
    Create a vector database instance
    
    Args:
        db_type (str): Type of vector database ('pinecone', 'chroma', 'qdrant')
        **kwargs: Additional arguments for the specific database
        
    Returns:
        VectorDBInterface: Vector database instance
    """
    if db_type.lower() == 'pinecone':
        return PineconeDB(**kwargs)
    elif db_type.lower() == 'chroma':
        return ChromaDB(**kwargs)
    elif db_type.lower() == 'qdrant':
        return QdrantDB(**kwargs)
    else:
        raise ValueError(f"Unsupported vector database type: {db_type}")


# Example usage
if __name__ == "__main__":
    # Use ChromaDB for local testing
    db = create_vector_db('chroma', persist_directory="./chroma_db")
    
    # Add documents
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on the use of data and algorithms to imitate the way that humans learn.",
        "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
        "Transformers are a type of neural network architecture used in NLP that can handle sequential data more effectively than previous models.",
        "Neural networks are computing systems inspired by biological neural networks that can learn to perform tasks by considering examples."
    ]
    
    metadata = [
        {"source": "ML", "author": "Author 1"},
        {"source": "ML", "author": "Author 2"},
        {"source": "ML", "author": "Author 3"},
        {"source": "ML", "author": "Author 4"}
    ]
    
    doc_ids = db.add_documents(documents, metadata, "ml_collection")
    print(f"Added documents with IDs: {doc_ids}")
    
    # Search
    query = "What is deep learning?"
    results, meta, scores = db.search(query, top_k=2, collection_name="ml_collection")
    
    print(f"Query: {query}")
    print("\nResults:")
    for i, (doc, metadata, score) in enumerate(zip(results, meta, scores)):
        print(f"{i+1}. ({score:.4f}) [{metadata.get('source', 'Unknown')}]: {doc[:100]}...")
