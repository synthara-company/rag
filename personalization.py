"""
Personalization module for the RAG Transformer system.
This module provides functionality to personalize responses based on user preferences and history.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
import faiss
from collections import Counter, defaultdict
import re
from datetime import datetime

class UserProfile:
    """Class for storing and managing user profile information"""
    
    def __init__(self, user_id: str, name: Optional[str] = None):
        """
        Initialize user profile
        
        Args:
            user_id (str): Unique identifier for the user
            name (Optional[str]): User's name
        """
        self.user_id = user_id
        self.name = name or user_id
        self.created_at = datetime.now().isoformat()
        self.last_active = self.created_at
        
        # Preferences
        self.interests = []
        self.preferred_sources = []
        self.topic_weights = {}
        
        # History
        self.query_history = []
        self.interaction_history = []
        
        # Embeddings
        self.interest_embeddings = []
    
    def update_activity(self) -> None:
        """Update last active timestamp"""
        self.last_active = datetime.now().isoformat()
    
    def add_interest(self, interest: str, weight: float = 1.0) -> None:
        """
        Add an interest to the user profile
        
        Args:
            interest (str): Interest topic
            weight (float): Weight of the interest (0-1)
        """
        # Check if interest already exists
        for i, (topic, _) in enumerate(self.interests):
            if topic.lower() == interest.lower():
                self.interests[i] = (topic, weight)
                return
        
        # Add new interest
        self.interests.append((interest, weight))
    
    def add_preferred_source(self, source: str, weight: float = 1.0) -> None:
        """
        Add a preferred source to the user profile
        
        Args:
            source (str): Source name
            weight (float): Weight of the preference (0-1)
        """
        # Check if source already exists
        for i, (src, _) in enumerate(self.preferred_sources):
            if src.lower() == source.lower():
                self.preferred_sources[i] = (src, weight)
                return
        
        # Add new source
        self.preferred_sources.append((source, weight))
    
    def add_query(self, query: str, timestamp: Optional[str] = None) -> None:
        """
        Add a query to the user's history
        
        Args:
            query (str): Query string
            timestamp (Optional[str]): Timestamp (ISO format)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        self.query_history.append({
            "query": query,
            "timestamp": timestamp
        })
        
        self.update_activity()
    
    def add_interaction(self, 
                        interaction_type: str, 
                        content: str, 
                        metadata: Optional[Dict[str, Any]] = None,
                        timestamp: Optional[str] = None) -> None:
        """
        Add an interaction to the user's history
        
        Args:
            interaction_type (str): Type of interaction (e.g., 'click', 'like', 'dislike')
            content (str): Content of the interaction
            metadata (Optional[Dict[str, Any]]): Additional metadata
            timestamp (Optional[str]): Timestamp (ISO format)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        if metadata is None:
            metadata = {}
        
        self.interaction_history.append({
            "type": interaction_type,
            "content": content,
            "metadata": metadata,
            "timestamp": timestamp
        })
        
        self.update_activity()
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, str]]:
        """
        Get the most recent queries
        
        Args:
            limit (int): Maximum number of queries to return
            
        Returns:
            List[Dict[str, str]]: Recent queries with timestamps
        """
        return sorted(
            self.query_history,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]
    
    def get_frequent_terms(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get the most frequently used terms in queries
        
        Args:
            limit (int): Maximum number of terms to return
            
        Returns:
            List[Tuple[str, int]]: Terms and their frequencies
        """
        # Extract terms from queries
        all_terms = []
        for query_data in self.query_history:
            query = query_data["query"].lower()
            terms = re.findall(r'\b[a-z]{3,}\b', query)  # Words with at least 3 letters
            all_terms.extend(terms)
        
        # Count term frequencies
        term_counts = Counter(all_terms)
        
        # Return most common terms
        return term_counts.most_common(limit)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert user profile to dictionary
        
        Returns:
            Dict[str, Any]: User profile as dictionary
        """
        return {
            "user_id": self.user_id,
            "name": self.name,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "interests": self.interests,
            "preferred_sources": self.preferred_sources,
            "topic_weights": self.topic_weights,
            "query_history": self.query_history,
            "interaction_history": self.interaction_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """
        Create user profile from dictionary
        
        Args:
            data (Dict[str, Any]): User profile data
            
        Returns:
            UserProfile: User profile instance
        """
        profile = cls(data["user_id"], data.get("name"))
        profile.created_at = data.get("created_at", profile.created_at)
        profile.last_active = data.get("last_active", profile.last_active)
        profile.interests = data.get("interests", [])
        profile.preferred_sources = data.get("preferred_sources", [])
        profile.topic_weights = data.get("topic_weights", {})
        profile.query_history = data.get("query_history", [])
        profile.interaction_history = data.get("interaction_history", [])
        
        return profile


class PersonalizationEngine:
    """Class for personalizing RAG responses based on user profiles"""
    
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 profiles_dir: str = 'user_profiles'):
        """
        Initialize personalization engine
        
        Args:
            embedding_model (str): Model for creating embeddings
            profiles_dir (str): Directory for storing user profiles
        """
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            print(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise
        
        # Set up profiles directory
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)
        
        # Load existing profiles
        self.profiles = {}
        self._load_profiles()
    
    def _load_profiles(self) -> None:
        """Load user profiles from disk"""
        try:
            for filename in os.listdir(self.profiles_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.profiles_dir, filename)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        profile = UserProfile.from_dict(data)
                        self.profiles[profile.user_id] = profile
            
            print(f"Loaded {len(self.profiles)} user profiles")
        except Exception as e:
            print(f"Error loading user profiles: {e}")
    
    def save_profile(self, profile: UserProfile) -> None:
        """
        Save user profile to disk
        
        Args:
            profile (UserProfile): User profile to save
        """
        try:
            file_path = os.path.join(self.profiles_dir, f"{profile.user_id}.json")
            with open(file_path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Error saving user profile: {e}")
    
    def get_profile(self, user_id: str) -> UserProfile:
        """
        Get user profile by ID
        
        Args:
            user_id (str): User ID
            
        Returns:
            UserProfile: User profile
        """
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id)
        
        return self.profiles[user_id]
    
    def update_profile_from_query(self, 
                                  user_id: str, 
                                  query: str, 
                                  response: Optional[Dict[str, Any]] = None) -> None:
        """
        Update user profile based on query and response
        
        Args:
            user_id (str): User ID
            query (str): Query string
            response (Optional[Dict[str, Any]]): Response data
        """
        profile = self.get_profile(user_id)
        
        # Add query to history
        profile.add_query(query)
        
        # Extract potential interests from query
        self._extract_interests_from_query(profile, query)
        
        # Update profile based on response
        if response:
            self._update_from_response(profile, response)
        
        # Save updated profile
        self.save_profile(profile)
    
    def _extract_interests_from_query(self, profile: UserProfile, query: str) -> None:
        """
        Extract potential interests from query
        
        Args:
            profile (UserProfile): User profile
            query (str): Query string
        """
        # Simple keyword extraction (could be improved with NLP)
        keywords = re.findall(r'\b[A-Za-z]{3,}\b', query)  # Words with at least 3 letters
        
        # Count keyword frequencies across all queries
        all_keywords = []
        for query_data in profile.query_history:
            q = query_data["query"]
            words = re.findall(r'\b[A-Za-z]{3,}\b', q)
            all_keywords.extend(words)
        
        keyword_counts = Counter(all_keywords)
        
        # Add frequent keywords as interests
        for keyword, count in keyword_counts.items():
            if count >= 3:  # If keyword appears at least 3 times
                weight = min(1.0, count / 10)  # Cap weight at 1.0
                profile.add_interest(keyword, weight)
    
    def _update_from_response(self, profile: UserProfile, response: Dict[str, Any]) -> None:
        """
        Update profile based on response
        
        Args:
            profile (UserProfile): User profile
            response (Dict[str, Any]): Response data
        """
        # Update preferred sources
        if 'text_sources' in response:
            source_counts = Counter(response['text_sources'])
            for source, count in source_counts.items():
                # Check if source already exists
                exists = False
                for i, (src, weight) in enumerate(profile.preferred_sources):
                    if src.lower() == source.lower():
                        # Update weight (slowly increase)
                        new_weight = min(1.0, weight + 0.1)
                        profile.preferred_sources[i] = (src, new_weight)
                        exists = True
                        break
                
                if not exists:
                    # Add new source with low initial weight
                    profile.add_preferred_source(source, 0.3)
    
    def personalize_retrieval(self,
                              user_id: str,
                              documents: List[str],
                              sources: List[str],
                              scores: List[float],
                              top_k: int) -> Tuple[List[str], List[str], List[float]]:
        """
        Personalize retrieval results based on user profile
        
        Args:
            user_id (str): User ID
            documents (List[str]): Retrieved documents
            sources (List[str]): Document sources
            scores (List[float]): Retrieval scores
            top_k (int): Number of documents to return
            
        Returns:
            Tuple[List[str], List[str], List[float]]:
                Personalized documents, sources, and scores
        """
        if not documents:
            return [], [], []
        
        profile = self.get_profile(user_id)
        
        # If no preferences, return original results
        if not profile.interests and not profile.preferred_sources:
            return documents[:top_k], sources[:top_k], scores[:top_k]
        
        # Adjust scores based on user preferences
        adjusted_scores = scores.copy()
        
        # Adjust based on preferred sources
        if profile.preferred_sources:
            source_weights = {src.lower(): weight for src, weight in profile.preferred_sources}
            
            for i, source in enumerate(sources):
                if source.lower() in source_weights:
                    # Boost score based on source preference
                    adjusted_scores[i] *= (1.0 + source_weights[source.lower()])
        
        # Adjust based on interests
        if profile.interests:
            # Get interest terms and weights
            interest_terms = [term.lower() for term, _ in profile.interests]
            interest_weights = {term.lower(): weight for term, weight in profile.interests}
            
            for i, doc in enumerate(documents):
                doc_lower = doc.lower()
                for term, weight in interest_weights.items():
                    if term in doc_lower:
                        # Boost score based on interest match
                        adjusted_scores[i] *= (1.0 + weight * 0.5)
        
        # Sort by adjusted scores
        sorted_indices = np.argsort(adjusted_scores)[::-1]
        
        # Get top-k results
        personalized_docs = [documents[i] for i in sorted_indices[:top_k]]
        personalized_sources = [sources[i] for i in sorted_indices[:top_k]]
        personalized_scores = [adjusted_scores[i] for i in sorted_indices[:top_k]]
        
        return personalized_docs, personalized_sources, personalized_scores
    
    def personalize_prompt(self, user_id: str, prompt: str) -> str:
        """
        Personalize prompt based on user profile
        
        Args:
            user_id (str): User ID
            prompt (str): Original prompt
            
        Returns:
            str: Personalized prompt
        """
        profile = self.get_profile(user_id)
        
        # If no preferences or history, return original prompt
        if not profile.interests and not profile.query_history:
            return prompt
        
        # Add user context to prompt
        user_context = f"User name: {profile.name}\n"
        
        # Add interests
        if profile.interests:
            user_context += "User interests: "
            interest_terms = [term for term, _ in profile.interests[:5]]  # Top 5 interests
            user_context += ", ".join(interest_terms)
            user_context += "\n"
        
        # Add recent queries
        recent_queries = profile.get_recent_queries(3)  # Last 3 queries
        if recent_queries:
            user_context += "Recent queries: "
            query_texts = [q["query"] for q in recent_queries]
            user_context += "; ".join(query_texts)
            user_context += "\n"
        
        # Insert user context into prompt
        personalized_prompt = f"""User context:
{user_context}

{prompt}"""
        
        return personalized_prompt
    
    def log_interaction(self,
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
        profile = self.get_profile(user_id)
        profile.add_interaction(interaction_type, content, metadata)
        self.save_profile(profile)
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about user activity
        
        Args:
            user_id (str): User ID
            
        Returns:
            Dict[str, Any]: User statistics
        """
        profile = self.get_profile(user_id)
        
        # Calculate statistics
        stats = {
            "user_id": profile.user_id,
            "name": profile.name,
            "created_at": profile.created_at,
            "last_active": profile.last_active,
            "total_queries": len(profile.query_history),
            "total_interactions": len(profile.interaction_history),
            "interests": profile.interests,
            "preferred_sources": profile.preferred_sources,
            "frequent_terms": profile.get_frequent_terms(10)
        }
        
        # Calculate interaction statistics
        interaction_types = [i["type"] for i in profile.interaction_history]
        interaction_counts = Counter(interaction_types)
        stats["interaction_counts"] = dict(interaction_counts)
        
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize personalization engine
    engine = PersonalizationEngine()
    
    # Create or get user profile
    user_id = "user123"
    profile = engine.get_profile(user_id)
    
    # Add some interests
    profile.add_interest("machine learning", 0.8)
    profile.add_interest("neural networks", 0.7)
    
    # Add some queries
    profile.add_query("What is deep learning?")
    profile.add_query("How do transformers work?")
    
    # Save profile
    engine.save_profile(profile)
    
    # Test personalization
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on the use of data and algorithms to imitate the way that humans learn.",
        "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
        "Transformers are a type of neural network architecture used in NLP that can handle sequential data more effectively than previous models.",
        "Supervised learning involves training a model on labeled data, where the desired output is known."
    ]
    
    sources = ["ML", "ML", "ML", "ML"]
    scores = [0.8, 0.7, 0.6, 0.5]
    
    # Personalize retrieval
    personalized_docs, personalized_sources, personalized_scores = engine.personalize_retrieval(
        user_id, documents, sources, scores, top_k=3
    )
    
    print("Original documents:")
    for i, (doc, score) in enumerate(zip(documents, scores)):
        print(f"{i+1}. ({score:.2f}): {doc[:50]}...")
    
    print("\nPersonalized documents:")
    for i, (doc, score) in enumerate(zip(personalized_docs, personalized_scores)):
        print(f"{i+1}. ({score:.2f}): {doc[:50]}...")
    
    # Personalize prompt
    original_prompt = "Generate a response about neural networks."
    personalized_prompt = engine.personalize_prompt(user_id, original_prompt)
    
    print("\nOriginal prompt:")
    print(original_prompt)
    
    print("\nPersonalized prompt:")
    print(personalized_prompt)
    
    # Get user stats
    stats = engine.get_user_stats(user_id)
    print("\nUser stats:")
    print(f"Total queries: {stats['total_queries']}")
    print(f"Interests: {stats['interests']}")
