"""
Knowledge graph visualization for the RAG Transformer system.
This module provides functionality to create and visualize knowledge graphs.
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple, Optional, Set
import spacy
import re
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class KnowledgeGraphBuilder:
    """Class for building knowledge graphs from documents"""
    
    def __init__(self, use_spacy: bool = True, embedding_model: Optional[str] = None):
        """
        Initialize knowledge graph builder
        
        Args:
            use_spacy (bool): Whether to use spaCy for entity extraction
            embedding_model (Optional[str]): Name of the embedding model to use
        """
        self.graph = nx.DiGraph()
        self.use_spacy = use_spacy
        
        # Load spaCy model if requested
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("Loaded spaCy model for entity extraction")
            except Exception as e:
                print(f"Error loading spaCy model: {e}")
                print("To install: python -m spacy download en_core_web_sm")
                self.use_spacy = False
        
        # Load embedding model if specified
        self.use_embeddings = embedding_model is not None
        if self.use_embeddings:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                self.use_embeddings = False
    
    def extract_entities_spacy(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from text using spaCy
        
        Args:
            text (str): Input text
            
        Returns:
            List[Tuple[str, str]]: List of (entity, entity_type) tuples
        """
        if not self.use_spacy:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Filter out very short entities and dates/times/percentages
            if len(ent.text) > 2 and ent.label_ not in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                entities.append((ent.text, ent.label_))
        
        return entities
    
    def extract_entities_regex(self, text: str) -> List[str]:
        """
        Extract potential entities using regex patterns
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of extracted entities
        """
        # Extract capitalized phrases (potential named entities)
        capitalized = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text)
        
        # Extract technical terms (lowercase words with specific endings)
        technical = re.findall(r'\b[a-z]+(?:ing|tion|ment|ity|ism|ology|ence|ance)\b', text)
        
        # Combine and filter
        all_entities = capitalized + technical
        filtered = [e for e in all_entities if len(e) > 3]  # Filter out very short entities
        
        return filtered
    
    def extract_relations(self, text: str, entities: List[str]) -> List[Tuple[str, str, float]]:
        """
        Extract potential relations between entities
        
        Args:
            text (str): Input text
            entities (List[str]): List of entities
            
        Returns:
            List[Tuple[str, str, float]]: List of (entity1, entity2, strength) tuples
        """
        relations = []
        
        # Simple co-occurrence based relation extraction
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Find entities in this sentence
            sentence_entities = [e for e in entities if e.lower() in sentence.lower()]
            
            # Create relations between all pairs of entities in the same sentence
            for i, e1 in enumerate(sentence_entities):
                for e2 in sentence_entities[i+1:]:
                    # Avoid self-relations
                    if e1 != e2:
                        relations.append((e1, e2, 1.0))
        
        return relations
    
    def compute_entity_embeddings(self, entities: List[str]) -> Dict[str, np.ndarray]:
        """
        Compute embeddings for entities
        
        Args:
            entities (List[str]): List of entities
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping entities to embeddings
        """
        if not self.use_embeddings:
            return {}
        
        embeddings = {}
        
        try:
            # Compute embeddings in batches
            batch_size = 32
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(batch)
                
                for j, entity in enumerate(batch):
                    embeddings[entity] = batch_embeddings[j]
        except Exception as e:
            print(f"Error computing embeddings: {e}")
        
        return embeddings
    
    def find_semantic_relations(self, entity_embeddings: Dict[str, np.ndarray], threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Find semantic relations between entities based on embedding similarity
        
        Args:
            entity_embeddings (Dict[str, np.ndarray]): Dictionary mapping entities to embeddings
            threshold (float): Similarity threshold
            
        Returns:
            List[Tuple[str, str, float]]: List of (entity1, entity2, similarity) tuples
        """
        relations = []
        
        entities = list(entity_embeddings.keys())
        
        if len(entities) < 2:
            return relations
        
        # Compute pairwise similarities
        embeddings = np.array([entity_embeddings[e] for e in entities])
        similarities = cosine_similarity(embeddings)
        
        # Extract relations above threshold
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                similarity = similarities[i, j]
                if similarity > threshold:
                    relations.append((entities[i], entities[j], float(similarity)))
        
        return relations
    
    def build_graph_from_documents(self, documents: List[str], sources: Optional[List[str]] = None) -> nx.DiGraph:
        """
        Build a knowledge graph from documents
        
        Args:
            documents (List[str]): List of documents
            sources (Optional[List[str]]): List of document sources
            
        Returns:
            nx.DiGraph: Knowledge graph
        """
        self.graph = nx.DiGraph()
        
        # Track all entities and their frequencies
        all_entities = []
        entity_sources = {}
        
        # Process each document
        for i, doc in enumerate(documents):
            source = sources[i] if sources and i < len(sources) else "Unknown"
            
            # Extract entities
            if self.use_spacy:
                spacy_entities = self.extract_entities_spacy(doc)
                entities = [e[0] for e in spacy_entities]
                entity_types = {e[0]: e[1] for e in spacy_entities}
            else:
                entities = self.extract_entities_regex(doc)
                entity_types = {e: "UNKNOWN" for e in entities}
            
            # Track entities and their sources
            all_entities.extend(entities)
            for entity in entities:
                if entity not in entity_sources:
                    entity_sources[entity] = set()
                entity_sources[entity].add(source)
            
            # Extract relations
            relations = self.extract_relations(doc, entities)
            
            # Add entities and relations to graph
            for entity in entities:
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, type=entity_types.get(entity, "UNKNOWN"), count=1, sources=set([source]))
                else:
                    self.graph.nodes[entity]["count"] += 1
                    self.graph.nodes[entity]["sources"].add(source)
            
            for e1, e2, strength in relations:
                if self.graph.has_edge(e1, e2):
                    self.graph.edges[e1, e2]["weight"] += strength
                else:
                    self.graph.add_edge(e1, e2, weight=strength, source=source)
        
        # Compute entity frequencies
        entity_counts = Counter(all_entities)
        
        # Compute entity embeddings if requested
        if self.use_embeddings:
            # Only compute embeddings for entities that appear multiple times
            frequent_entities = [e for e, c in entity_counts.items() if c > 1]
            entity_embeddings = self.compute_entity_embeddings(frequent_entities)
            
            # Find semantic relations
            semantic_relations = self.find_semantic_relations(entity_embeddings)
            
            # Add semantic relations to graph
            for e1, e2, similarity in semantic_relations:
                if not self.graph.has_edge(e1, e2) and not self.graph.has_edge(e2, e1):
                    self.graph.add_edge(e1, e2, weight=similarity, type="semantic")
        
        # Update node attributes with frequency and source information
        for entity, count in entity_counts.items():
            if self.graph.has_node(entity):
                self.graph.nodes[entity]["frequency"] = count
                self.graph.nodes[entity]["sources"] = list(entity_sources.get(entity, []))
        
        return self.graph
    
    def prune_graph(self, min_edge_weight: float = 0.5, min_node_frequency: int = 2, max_nodes: int = 100) -> nx.DiGraph:
        """
        Prune the knowledge graph to remove weak connections and infrequent entities
        
        Args:
            min_edge_weight (float): Minimum edge weight to keep
            min_node_frequency (int): Minimum node frequency to keep
            max_nodes (int): Maximum number of nodes to keep
            
        Returns:
            nx.DiGraph: Pruned knowledge graph
        """
        # Create a copy of the graph
        pruned_graph = self.graph.copy()
        
        # Remove weak edges
        weak_edges = [(u, v) for u, v, d in pruned_graph.edges(data=True) if d.get("weight", 0) < min_edge_weight]
        pruned_graph.remove_edges_from(weak_edges)
        
        # Remove infrequent nodes
        infrequent_nodes = [n for n, d in pruned_graph.nodes(data=True) if d.get("frequency", 0) < min_node_frequency]
        pruned_graph.remove_nodes_from(infrequent_nodes)
        
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(pruned_graph))
        pruned_graph.remove_nodes_from(isolated_nodes)
        
        # If still too many nodes, keep only the most frequent ones
        if pruned_graph.number_of_nodes() > max_nodes:
            node_frequencies = {n: d.get("frequency", 0) for n, d in pruned_graph.nodes(data=True)}
            nodes_to_keep = sorted(node_frequencies.keys(), key=lambda x: node_frequencies[x], reverse=True)[:max_nodes]
            nodes_to_remove = [n for n in pruned_graph.nodes() if n not in nodes_to_keep]
            pruned_graph.remove_nodes_from(nodes_to_remove)
        
        return pruned_graph
    
    def get_community_structure(self, graph: Optional[nx.DiGraph] = None) -> Dict[str, int]:
        """
        Detect communities in the knowledge graph
        
        Args:
            graph (Optional[nx.DiGraph]): Graph to analyze (uses self.graph if None)
            
        Returns:
            Dict[str, int]: Dictionary mapping nodes to community IDs
        """
        if graph is None:
            graph = self.graph
        
        # Convert to undirected graph for community detection
        undirected = graph.to_undirected()
        
        # Detect communities using Louvain algorithm
        try:
            from community import best_partition
            partition = best_partition(undirected)
        except ImportError:
            print("Community detection requires python-louvain package. Install with 'pip install python-louvain'")
            # Fallback: assign random communities
            partition = {node: hash(node) % 10 for node in undirected.nodes()}
        
        return partition


class KnowledgeGraphVisualizer:
    """Class for visualizing knowledge graphs"""
    
    @staticmethod
    def visualize_matplotlib(graph: nx.DiGraph, output_path: Optional[str] = None, title: str = "Knowledge Graph") -> plt.Figure:
        """
        Visualize knowledge graph using matplotlib
        
        Args:
            graph (nx.DiGraph): Knowledge graph
            output_path (Optional[str]): Path to save the visualization
            title (str): Title for the visualization
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        plt.figure(figsize=(20, 15))
        
        # Node sizes based on frequency or degree
        node_sizes = [300 * (1 + graph.degree(n)) for n in graph.nodes()]
        
        # Node colors based on type or community
        if all("type" in graph.nodes[n] for n in graph.nodes()):
            node_types = [graph.nodes[n]["type"] for n in graph.nodes()]
            unique_types = list(set(node_types))
            color_map = plt.cm.get_cmap("tab20", len(unique_types))
            node_colors = [color_map(unique_types.index(t)) for t in node_types]
        else:
            # Default: color by degree
            node_colors = [0.1 + 0.8 * (graph.degree(n) / max(1, max(dict(graph.degree()).values()))) for n in graph.nodes()]
        
        # Edge weights
        edge_weights = [d.get("weight", 1.0) for u, v, d in graph.edges(data=True)]
        
        # Layout
        pos = nx.spring_layout(graph, k=0.3, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            graph, 
            pos, 
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            graph, 
            pos, 
            width=[w * 2 for w in edge_weights],
            alpha=0.5,
            arrows=True,
            arrowsize=15
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            graph, 
            pos, 
            font_size=10,
            font_weight="bold"
        )
        
        plt.title(title, fontsize=20)
        plt.axis("off")
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        
        return plt.gcf()
    
    @staticmethod
    def visualize_plotly(graph: nx.DiGraph, title: str = "Knowledge Graph") -> go.Figure:
        """
        Visualize knowledge graph using Plotly
        
        Args:
            graph (nx.DiGraph): Knowledge graph
            title (str): Title for the visualization
            
        Returns:
            go.Figure: Plotly figure
        """
        # Get node positions using spring layout
        pos = nx.spring_layout(graph, dim=3, k=0.3, iterations=50)
        
        # Extract node positions
        x_nodes = [pos[node][0] for node in graph.nodes()]
        y_nodes = [pos[node][1] for node in graph.nodes()]
        z_nodes = [pos[node][2] for node in graph.nodes()]
        
        # Node sizes based on frequency or degree
        node_sizes = [5 + 10 * graph.degree(n) for n in graph.nodes()]
        
        # Node colors based on type or community
        if all("type" in graph.nodes[n] for n in graph.nodes()):
            node_types = [graph.nodes[n]["type"] for n in graph.nodes()]
            unique_types = list(set(node_types))
            node_colors = [unique_types.index(t) for t in node_types]
        else:
            # Default: color by degree
            node_colors = [graph.degree(n) for n in graph.nodes()]
        
        # Create node trace
        node_trace = go.Scatter3d(
            x=x_nodes,
            y=y_nodes,
            z=z_nodes,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                opacity=0.8
            ),
            text=[f"Node: {n}<br>Degree: {graph.degree(n)}" for n in graph.nodes()],
            hoverinfo='text'
        )
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge in graph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            
            # Add line coordinates
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(
                color='rgba(50, 50, 50, 0.2)',
                width=1
            ),
            hoverinfo='none'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=False,
            scene=dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title='')
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig
    
    @staticmethod
    def export_graph_json(graph: nx.DiGraph, output_path: str) -> Dict[str, Any]:
        """
        Export knowledge graph to JSON format
        
        Args:
            graph (nx.DiGraph): Knowledge graph
            output_path (str): Path to save the JSON file
            
        Returns:
            Dict[str, Any]: Graph data in JSON format
        """
        # Convert graph to JSON-serializable format
        data = {
            "nodes": [],
            "links": []
        }
        
        # Add nodes
        for node, attrs in graph.nodes(data=True):
            node_data = {
                "id": node,
                "label": node
            }
            
            # Add node attributes
            for key, value in attrs.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    node_data[key] = value
                elif isinstance(value, (list, set)):
                    node_data[key] = list(value)
            
            data["nodes"].append(node_data)
        
        # Add edges
        for source, target, attrs in graph.edges(data=True):
            edge_data = {
                "source": source,
                "target": target
            }
            
            # Add edge attributes
            for key, value in attrs.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    edge_data[key] = value
            
            data["links"].append(edge_data)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return data


# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on the use of data and algorithms to imitate the way that humans learn.",
        "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
        "Transformers are a type of neural network architecture used in NLP that can handle sequential data more effectively than previous models.",
        "Neural networks are computing systems inspired by biological neural networks that can learn to perform tasks by considering examples."
    ]
    
    sources = ["ML", "ML", "ML", "ML"]
    
    # Build knowledge graph
    builder = KnowledgeGraphBuilder(use_spacy=True, embedding_model="all-MiniLM-L6-v2")
    graph = builder.build_graph_from_documents(documents, sources)
    
    # Prune graph
    pruned_graph = builder.prune_graph(min_edge_weight=0.5, min_node_frequency=1)
    
    # Visualize graph
    visualizer = KnowledgeGraphVisualizer()
    fig = visualizer.visualize_matplotlib(pruned_graph, "knowledge_graph_ml.png", "Machine Learning Knowledge Graph")
    
    # Export graph
    visualizer.export_graph_json(pruned_graph, "knowledge_graph_ml.json")
