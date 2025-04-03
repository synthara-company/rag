"""
Hybrid retrieval system for the RAG Transformer.
This module combines dense and sparse retrieval methods for better document retrieval.
"""

import os
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
import time
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
# Simple tokenizer and stopwords instead of nltk
import string

# Simple English stopwords list
ENGLISH_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
    'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such',
    'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during', 'to', 'from',
    'in', 'on', 'by', 'at', 'into', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'
}

class HybridRetriever:
    """Class for hybrid document retrieval combining dense and sparse methods"""

    def __init__(self,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 use_bm25: bool = True,
                 use_tfidf: bool = True,
                 alpha: float = 0.5):
        """
        Initialize hybrid retriever

        Args:
            embedding_model (str): Model for creating document embeddings
            use_bm25 (bool): Whether to use BM25 for sparse retrieval
            use_tfidf (bool): Whether to use TF-IDF for sparse retrieval
            alpha (float): Weight for combining dense and sparse scores (0-1)
                           Higher values give more weight to dense retrieval
        """
        self.alpha = alpha
        self.use_bm25 = use_bm25
        self.use_tfidf = use_tfidf

        # Initialize dense retrieval components
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            print(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise

        # Initialize sparse retrieval components
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2
        ) if use_tfidf else None

        self.bm25 = None  # Will be initialized when documents are added
        # Simple stemmer function instead of PorterStemmer
        self.stemmer = lambda word: word.lower().rstrip('s').rstrip('ed').rstrip('ing')
        self.stop_words = ENGLISH_STOPWORDS

        # Storage for documents and embeddings
        self.documents = []
        self.document_sources = []
        self.document_metadata = []
        self.dense_index = None
        self.tfidf_matrix = None
        self.tokenized_docs = []

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for sparse retrieval

        Args:
            text (str): Input text

        Returns:
            List[str]: Tokenized and preprocessed text
        """
        # Lowercase and tokenize (simple split by space and punctuation)
        text = text.lower()
        # Remove punctuation
        for char in string.punctuation:
            text = text.replace(char, ' ')
        tokens = text.split()

        # Remove stopwords and apply simple stemming
        tokens = [
            self.stemmer(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]

        return tokens

    def add_documents(self,
                      documents: List[str],
                      sources: Optional[List[str]] = None,
                      metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add documents to the retrieval system

        Args:
            documents (List[str]): List of documents
            sources (Optional[List[str]]): List of document sources
            metadata (Optional[List[Dict[str, Any]]]): List of document metadata
        """
        if not documents:
            return

        # Store documents
        self.documents.extend(documents)

        # Store sources
        if sources:
            if len(sources) != len(documents):
                sources = [sources[0] if sources else "Unknown"] * len(documents)
            self.document_sources.extend(sources)
        else:
            self.document_sources.extend(["Unknown"] * len(documents))

        # Store metadata
        if metadata:
            if len(metadata) != len(documents):
                metadata = [metadata[0] if metadata else {}] * len(documents)
            self.document_metadata.extend(metadata)
        else:
            self.document_metadata.extend([{}] * len(documents))

        # Rebuild indexes
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build dense and sparse indexes"""
        # Build dense index
        self._build_dense_index()

        # Build sparse indexes
        self._build_sparse_indexes()

    def _build_dense_index(self) -> None:
        """Build dense index using FAISS"""
        if not self.documents:
            print("No documents to index")
            return

        # Embed documents
        embeddings = self.embedding_model.encode(self.documents)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.dense_index = faiss.IndexFlatL2(dimension)
        self.dense_index.add(embeddings)

        print(f"Built dense index with {len(self.documents)} documents")

    def _build_sparse_indexes(self) -> None:
        """Build sparse indexes (TF-IDF and BM25)"""
        if not self.documents:
            print("No documents to index")
            return

        # Build TF-IDF index
        if self.use_tfidf:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
            print(f"Built TF-IDF index with {self.tfidf_matrix.shape[1]} features")

        # Build BM25 index
        if self.use_bm25:
            # Tokenize documents
            self.tokenized_docs = [self.preprocess_text(doc) for doc in self.documents]
            self.bm25 = BM25Okapi(self.tokenized_docs)
            print(f"Built BM25 index with {len(self.tokenized_docs)} documents")

    def retrieve(self,
                 query: str,
                 top_k: int = 5,
                 alpha: Optional[float] = None) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float], Dict[str, Any]]:
        """
        Retrieve documents using hybrid approach

        Args:
            query (str): Query string
            top_k (int): Number of documents to retrieve
            alpha (Optional[float]): Weight for combining dense and sparse scores
                                    If None, use the instance's alpha value

        Returns:
            Tuple[List[str], List[str], List[Dict[str, Any]], List[float], Dict[str, Any]]:
                Retrieved documents, their sources, metadata, scores, and retrieval details
        """
        if not self.documents:
            return [], [], [], [], {"error": "No documents in the index"}

        if alpha is None:
            alpha = self.alpha

        start_time = time.time()
        retrieval_details = {
            "query": query,
            "top_k": top_k,
            "alpha": alpha,
            "methods": []
        }

        # Get dense retrieval results
        dense_start = time.time()
        dense_docs, dense_scores, dense_indices = self._dense_retrieval(query, top_k * 2)
        dense_time = time.time() - dense_start
        retrieval_details["methods"].append({
            "name": "dense",
            "model": self.embedding_model.__class__.__name__,
            "time": dense_time
        })

        # Get sparse retrieval results
        sparse_docs = []
        sparse_scores = []
        sparse_indices = []

        if self.use_tfidf or self.use_bm25:
            sparse_start = time.time()
            sparse_docs, sparse_scores, sparse_indices = self._sparse_retrieval(query, top_k * 2)
            sparse_time = time.time() - sparse_start
            retrieval_details["methods"].append({
                "name": "sparse",
                "models": [
                    "TF-IDF" if self.use_tfidf else None,
                    "BM25" if self.use_bm25 else None
                ],
                "time": sparse_time
            })

        # Combine results
        if sparse_docs and dense_docs:
            hybrid_start = time.time()
            combined_docs, combined_sources, combined_metadata, combined_scores = self._combine_results(
                dense_docs, dense_scores, dense_indices,
                sparse_docs, sparse_scores, sparse_indices,
                alpha, top_k
            )
            hybrid_time = time.time() - hybrid_start
            retrieval_details["methods"].append({
                "name": "hybrid",
                "alpha": alpha,
                "time": hybrid_time
            })
        else:
            # If one method failed, use the other
            if dense_docs:
                combined_docs = dense_docs[:top_k]
                combined_scores = dense_scores[:top_k]
                combined_indices = dense_indices[:top_k]
            else:
                combined_docs = sparse_docs[:top_k]
                combined_scores = sparse_scores[:top_k]
                combined_indices = sparse_indices[:top_k]

            # Get sources and metadata
            combined_sources = [self.document_sources[i] for i in combined_indices]
            combined_metadata = [self.document_metadata[i] for i in combined_indices]

        total_time = time.time() - start_time
        retrieval_details["total_time"] = total_time

        return combined_docs, combined_sources, combined_metadata, combined_scores, retrieval_details

    def _dense_retrieval(self, query: str, top_k: int) -> Tuple[List[str], List[float], List[int]]:
        """
        Perform dense retrieval using FAISS

        Args:
            query (str): Query string
            top_k (int): Number of documents to retrieve

        Returns:
            Tuple[List[str], List[float], List[int]]:
                Retrieved documents, their scores, and their indices
        """
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])

            # Search in FAISS index
            distances, indices = self.dense_index.search(query_embedding, min(top_k, len(self.documents)))

            # Convert distances to similarity scores (lower distance = higher similarity)
            max_dist = np.max(distances[0]) if distances[0].size > 0 else 1.0
            similarity_scores = [1.0 - (dist / max_dist) for dist in distances[0]]

            # Get documents
            retrieved_docs = [self.documents[idx] for idx in indices[0]]
            retrieved_indices = [int(idx) for idx in indices[0]]

            return retrieved_docs, similarity_scores, retrieved_indices
        except Exception as e:
            print(f"Error in dense retrieval: {e}")
            return [], [], []

    def _sparse_retrieval(self, query: str, top_k: int) -> Tuple[List[str], List[float], List[int]]:
        """
        Perform sparse retrieval using TF-IDF and/or BM25

        Args:
            query (str): Query string
            top_k (int): Number of documents to retrieve

        Returns:
            Tuple[List[str], List[float], List[int]]:
                Retrieved documents, their scores, and their indices
        """
        tfidf_scores = []
        bm25_scores = []

        # Get TF-IDF scores
        if self.use_tfidf and self.tfidf_matrix is not None:
            try:
                # Transform query
                query_vector = self.tfidf_vectorizer.transform([query])

                # Calculate similarity scores
                tfidf_scores = (self.tfidf_matrix @ query_vector.T).toarray().flatten()
            except Exception as e:
                print(f"Error in TF-IDF retrieval: {e}")

        # Get BM25 scores
        if self.use_bm25 and self.bm25 is not None:
            try:
                # Tokenize query
                tokenized_query = self.preprocess_text(query)

                # Get BM25 scores
                bm25_scores = self.bm25.get_scores(tokenized_query)
            except Exception as e:
                print(f"Error in BM25 retrieval: {e}")

        # Combine sparse scores
        if len(tfidf_scores) > 0 and len(bm25_scores) > 0:
            # Normalize scores
            tfidf_scores = tfidf_scores / (np.max(tfidf_scores) or 1.0)
            bm25_scores = bm25_scores / (np.max(bm25_scores) or 1.0)

            # Average scores
            sparse_scores = (tfidf_scores + bm25_scores) / 2
        elif len(tfidf_scores) > 0:
            sparse_scores = tfidf_scores / (np.max(tfidf_scores) or 1.0)
        elif len(bm25_scores) > 0:
            sparse_scores = bm25_scores / (np.max(bm25_scores) or 1.0)
        else:
            return [], [], []

        # Get top-k indices
        top_indices = np.argsort(sparse_scores)[::-1][:top_k]

        # Get documents and scores
        retrieved_docs = [self.documents[idx] for idx in top_indices]
        retrieved_scores = [float(sparse_scores[idx]) for idx in top_indices]
        retrieved_indices = [int(idx) for idx in top_indices]

        return retrieved_docs, retrieved_scores, retrieved_indices

    def _combine_results(self,
                         dense_docs: List[str],
                         dense_scores: List[float],
                         dense_indices: List[int],
                         sparse_docs: List[str],
                         sparse_scores: List[float],
                         sparse_indices: List[int],
                         alpha: float,
                         top_k: int) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
        """
        Combine dense and sparse retrieval results

        Args:
            dense_docs (List[str]): Documents from dense retrieval
            dense_scores (List[float]): Scores from dense retrieval
            dense_indices (List[int]): Indices from dense retrieval
            sparse_docs (List[str]): Documents from sparse retrieval
            sparse_scores (List[float]): Scores from sparse retrieval
            sparse_indices (List[int]): Indices from sparse retrieval
            alpha (float): Weight for combining scores (0-1)
            top_k (int): Number of documents to return

        Returns:
            Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
                Combined documents, their sources, metadata, and scores
        """
        # Create a mapping of document index to scores
        combined_scores = {}

        # Add dense scores
        for idx, score in zip(dense_indices, dense_scores):
            combined_scores[idx] = alpha * score

        # Add sparse scores
        for idx, score in zip(sparse_indices, sparse_scores):
            if idx in combined_scores:
                combined_scores[idx] += (1 - alpha) * score
            else:
                combined_scores[idx] = (1 - alpha) * score

        # Sort by score
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Get top-k items
        top_indices = [idx for idx, _ in sorted_items[:top_k]]
        top_scores = [score for _, score in sorted_items[:top_k]]

        # Get documents, sources, and metadata
        top_docs = [self.documents[idx] for idx in top_indices]
        top_sources = [self.document_sources[idx] for idx in top_indices]
        top_metadata = [self.document_metadata[idx] for idx in top_indices]

        return top_docs, top_sources, top_metadata, top_scores


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on the use of data and algorithms to imitate the way that humans learn.",
        "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
        "Transformers are a type of neural network architecture used in NLP that can handle sequential data more effectively than previous models.",
        "Neural networks are computing systems inspired by biological neural networks that can learn to perform tasks by considering examples.",
        "Supervised learning involves training a model on labeled data, where the desired output is known.",
        "Unsupervised learning finds hidden patterns or intrinsic structures in input data without labeled responses.",
        "Reinforcement learning is about training an agent to make decisions by rewarding desired behaviors and punishing undesired ones.",
        "Feature extraction is a key process in machine learning where important characteristics are identified and selected from raw data.",
        "Overfitting occurs when a machine learning model learns the training data too well, including its noise and fluctuations."
    ]

    sources = ["ML"] * len(documents)

    # Initialize hybrid retriever
    retriever = HybridRetriever(use_bm25=True, use_tfidf=True, alpha=0.7)

    # Add documents
    retriever.add_documents(documents, sources)

    # Test retrieval
    query = "What is deep learning?"
    docs, sources, metadata, scores, details = retriever.retrieve(query, top_k=3)

    print(f"Query: {query}")
    print(f"Retrieval details: {details}")
    print("\nRetrieved documents:")
    for i, (doc, source, score) in enumerate(zip(docs, sources, scores)):
        print(f"{i+1}. [{source}] ({score:.4f}): {doc[:100]}...")
