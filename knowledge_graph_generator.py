import os
import networkx as nx
import matplotlib.pyplot as plt
import json

class RAGKnowledgeGraphGenerator:
    def __init__(self):
        """
        Initialize the knowledge graph generator
        """
        self.graph = nx.DiGraph()
        
    def load_datasets(self, dataset_dir='datasets'):
        """
        Load datasets to extract knowledge domains
        
        Args:
            dataset_dir (str): Directory containing dataset JSON files
        """
        # Paths to datasets
        sci_fi_movies_path = os.path.join(dataset_dir, 'sci_fi_movies.json')
        cosmos_content_path = os.path.join(dataset_dir, 'cosmos_content.json')
        
        # Machine Learning Core Concepts
        ml_concepts = [
            "Supervised Learning", 
            "Unsupervised Learning", 
            "Reinforcement Learning", 
            "Deep Learning", 
            "Neural Networks", 
            "Transformers"
        ]
        
        # Sci-Fi Movie Genres
        sci_fi_genres = []
        try:
            with open(sci_fi_movies_path, 'r') as f:
                sci_fi_movies = json.load(f)
                sci_fi_genres = list(set([
                    genre 
                    for movie in sci_fi_movies 
                    for genre in movie.get('genres', [])
                ]))
        except FileNotFoundError:
            print(f"Sci-fi movies dataset not found at {sci_fi_movies_path}")
        
        # Cosmos Content Types
        cosmos_types = []
        try:
            with open(cosmos_content_path, 'r') as f:
                cosmos_content = json.load(f)
                cosmos_types = list(set([
                    content.get('media_type', 'Unknown') 
                    for content in cosmos_content
                ]))
        except FileNotFoundError:
            print(f"Cosmos content dataset not found at {cosmos_content_path}")
        
        return ml_concepts, sci_fi_genres, cosmos_types
    
    def create_knowledge_graph(self):
        """
        Create a comprehensive knowledge graph
        """
        # Load datasets
        ml_concepts, sci_fi_genres, cosmos_types = self.load_datasets()
        
        # Root Node
        self.graph.add_node("RAG Transformer", color='red', size=3000)
        
        # Knowledge Domains
        domains = {
            "Machine Learning": ml_concepts,
            "Science Fiction": sci_fi_genres,
            "Cosmos": cosmos_types
        }
        
        # Add Domain Nodes
        domain_colors = {
            "Machine Learning": 'blue', 
            "Science Fiction": 'green', 
            "Cosmos": 'purple'
        }
        
        for domain, concepts in domains.items():
            # Add domain node
            self.graph.add_node(domain, color=domain_colors[domain], size=2000)
            self.graph.add_edge("RAG Transformer", domain)
            
            # Add concept nodes
            for concept in concepts:
                self.graph.add_node(concept, color=domain_colors[domain], size=1000)
                self.graph.add_edge(domain, concept)
        
        # Technical Components
        tech_components = [
            "TMDB API", 
            "NASA API", 
            "Hugging Face Transformers", 
            "FAISS Index", 
            "Sentence Embeddings"
        ]
        
        tech_node = "Technical Components"
        self.graph.add_node(tech_node, color='orange', size=2000)
        self.graph.add_edge("RAG Transformer", tech_node)
        
        for component in tech_components:
            self.graph.add_node(component, color='orange', size=1000)
            self.graph.add_edge(tech_node, component)
    
    def visualize_graph(self, output_path='knowledge_graph.png'):
        """
        Visualize the knowledge graph
        
        Args:
            output_path (str): Path to save the graph visualization
        """
        plt.figure(figsize=(20, 15))
        
        # Get node attributes
        node_colors = [self.graph.nodes[node].get('color', 'gray') for node in self.graph.nodes()]
        node_sizes = [self.graph.nodes[node].get('size', 1000) for node in self.graph.nodes()]
        
        # Use spring layout for organic graph arrangement
        pos = nx.spring_layout(self.graph, k=0.5)  # k regulates the distance between nodes
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, 
            pos, 
            node_color=node_colors, 
            node_size=node_sizes, 
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, 
            pos, 
            edge_color='gray', 
            arrows=True, 
            alpha=0.3
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph, 
            pos, 
            font_size=10, 
            font_weight="bold"
        )
        
        plt.title("RAG Transformer Knowledge Graph", fontsize=20)
        plt.axis('off')
        
        # Save the graph
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Knowledge graph saved to {output_path}")

def main():
    # Create knowledge graph generator
    graph_generator = RAGKnowledgeGraphGenerator()
    
    # Create the graph
    graph_generator.create_knowledge_graph()
    
    # Visualize and save
    graph_generator.visualize_graph()

if __name__ == "__main__":
    main()
