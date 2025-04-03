"""
Command-line interface for the RAG System.
This is a very simple version that runs in the terminal.
"""

from simple_rag_system import SimpleRAGSystem
import time

def main():
    print("\n===== RAG Command Line Interface =====\n")
    print("Loading models and documents...")
    
    # Initialize RAG system
    rag_system = SimpleRAGSystem()
    
    # Load sample documents
    doc_count = rag_system.load_sample_documents("sample_documents.json")
    print(f"Loaded {doc_count} documents\n")
    
    print("Ready! Type your questions or commands below.")
    print("Commands: 'exit' to quit, 'add' to add a document, 'stats' to see statistics\n")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for commands
        if user_input.lower() == 'exit':
            print("\nGoodbye!")
            break
        
        elif user_input.lower() == 'add':
            # Add a document
            doc_text = input("\nEnter document text: ").strip()
            if not doc_text:
                print("Document text cannot be empty.")
                continue
                
            doc_source = input("Enter source (default: Custom): ").strip() or "Custom"
            
            rag_system.add_documents([doc_text], [doc_source])
            print(f"\nAdded document from source '{doc_source}'")
            continue
        
        elif user_input.lower() == 'stats':
            # Show statistics
            metrics = rag_system.get_performance_metrics()
            print("\n----- System Statistics -----")
            print(f"Documents: {metrics['document_count']}")
            print(f"Queries: {metrics['total_queries']}")
            if metrics['total_queries'] > 0:
                print(f"Avg. Retrieval Time: {metrics['avg_retrieval_time']:.4f}s")
                print(f"Avg. Generation Time: {metrics['avg_generation_time']:.4f}s")
                print(f"Avg. Total Time: {metrics['avg_total_time']:.4f}s")
            print("----------------------------")
            continue
        
        # Process query
        print("\nProcessing...")
        start_time = time.time()
        
        result = rag_system.process_query(user_input)
        
        total_time = time.time() - start_time
        
        # Display response
        print(f"\nAssistant: {result['response']}")
        
        # Display sources
        sources = {}
        for source in result['sources']:
            if source in sources:
                sources[source] += 1
            else:
                sources[source] = 1
        
        source_str = ", ".join([f"{source} ({count})" for source, count in sources.items()])
        print(f"\nSources: {source_str}")
        
        # Display timing information
        print(f"Time: {total_time:.2f}s (Retrieval: {result['retrieval_time']:.2f}s, Generation: {result['generation_time']:.2f}s)")
        
        # Ask if user wants to see documents
        show_docs = input("\nShow retrieved documents? (y/n): ").strip().lower()
        if show_docs == 'y':
            print("\n----- Retrieved Documents -----")
            for i, (doc, source, score) in enumerate(zip(result['documents'], result['sources'], result['scores'])):
                print(f"\nDocument {i+1} from {source} (Score: {score:.2f}):")
                # Show a preview of the document (first 300 characters)
                preview = doc[:300] + "..." if len(doc) > 300 else doc
                print(preview)
            print("------------------------------")

if __name__ == "__main__":
    main()
