"""
Synthara: An enhanced command-line interface for the RAG System.
Features ASCII art and improved visual formatting.
"""

from simple_rag_system import SimpleRAGSystem
import time
import os
import textwrap

# ANSI color codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_logo():
    """Print the Synthara ASCII logo"""
    logo = f"""{Colors.CYAN}
   _____             _   _                      
  / ____|           | | | |                     
 | (___  _   _ _ __ | |_| |__   __ _ _ __ __ _ 
  \\___ \\| | | | '_ \\| __| '_ \\ / _` | '__/ _` |
  ____) | |_| | | | | |_| | | | (_| | | | (_| |
 |_____/ \\__, |_| |_|\\__|_| |_|\\__,_|_|  \\__,_|
          __/ |                                 
         |___/                                  
{Colors.ENDC}"""
    print(logo)
    print(f"{Colors.BOLD}Retrieval-Augmented Generation Assistant{Colors.ENDC}")
    print(f"{Colors.YELLOW}{'=' * 50}{Colors.ENDC}\n")

def format_response(text):
    """Format the response text with proper wrapping"""
    wrapper = textwrap.TextWrapper(width=80, initial_indent="  ", subsequent_indent="  ")
    return wrapper.fill(text)

def print_section_header(title):
    """Print a formatted section header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * len(title)}{Colors.ENDC}")

def print_document(index, doc, source, score):
    """Print a formatted document"""
    print(f"\n{Colors.BOLD}Document {index+1}{Colors.ENDC} from {Colors.GREEN}{source}{Colors.ENDC} (Score: {Colors.YELLOW}{score:.2f}{Colors.ENDC}):")
    
    # Wrap and indent the document text
    wrapper = textwrap.TextWrapper(width=80, initial_indent="  ", subsequent_indent="  ")
    # Show a preview of the document (first 300 characters)
    preview = doc[:300] + "..." if len(doc) > 300 else doc
    print(wrapper.fill(preview))

def main():
    clear_screen()
    print_logo()
    
    print("Loading models and documents...")
    
    # Initialize RAG system
    rag_system = SimpleRAGSystem()
    
    # Load sample documents
    doc_count = rag_system.load_sample_documents("sample_documents.json")
    print(f"{Colors.GREEN}✓ Loaded {doc_count} documents{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Ready! Type your questions or commands below.{Colors.ENDC}")
    print("Commands:")
    print(f"  {Colors.CYAN}exit{Colors.ENDC} - Quit the application")
    print(f"  {Colors.CYAN}add{Colors.ENDC}  - Add a new document")
    print(f"  {Colors.CYAN}stats{Colors.ENDC} - View system statistics")
    print(f"  {Colors.CYAN}clear{Colors.ENDC} - Clear the screen\n")
    
    while True:
        # Get user input
        user_input = input(f"\n{Colors.BOLD}You:{Colors.ENDC} ").strip()
        
        # Check for commands
        if user_input.lower() == 'exit':
            print(f"\n{Colors.GREEN}Thank you for using Synthara! Goodbye!{Colors.ENDC}")
            break
        
        elif user_input.lower() == 'clear':
            clear_screen()
            print_logo()
            continue
        
        elif user_input.lower() == 'add':
            # Add a document
            print_section_header("Add New Document")
            doc_text = input(f"\nEnter document text: ").strip()
            if not doc_text:
                print(f"{Colors.RED}Document text cannot be empty.{Colors.ENDC}")
                continue
                
            doc_source = input("Enter source (default: Custom): ").strip() or "Custom"
            
            rag_system.add_documents([doc_text], [doc_source])
            print(f"\n{Colors.GREEN}✓ Added document from source '{doc_source}'{Colors.ENDC}")
            continue
        
        elif user_input.lower() == 'stats':
            # Show statistics
            metrics = rag_system.get_performance_metrics()
            print_section_header("System Statistics")
            print(f"Documents: {Colors.BOLD}{metrics['document_count']}{Colors.ENDC}")
            print(f"Queries: {Colors.BOLD}{metrics['total_queries']}{Colors.ENDC}")
            if metrics['total_queries'] > 0:
                print(f"Avg. Retrieval Time: {Colors.YELLOW}{metrics['avg_retrieval_time']:.4f}s{Colors.ENDC}")
                print(f"Avg. Generation Time: {Colors.YELLOW}{metrics['avg_generation_time']:.4f}s{Colors.ENDC}")
                print(f"Avg. Total Time: {Colors.YELLOW}{metrics['avg_total_time']:.4f}s{Colors.ENDC}")
            continue
        
        # Process query
        print(f"\n{Colors.YELLOW}Processing...{Colors.ENDC}")
        start_time = time.time()
        
        result = rag_system.process_query(user_input)
        
        total_time = time.time() - start_time
        
        # Display response
        print(f"\n{Colors.BOLD}Synthara:{Colors.ENDC} {Colors.CYAN}{format_response(result['response'])}{Colors.ENDC}")
        
        # Display sources
        sources = {}
        for source in result['sources']:
            if source in sources:
                sources[source] += 1
            else:
                sources[source] = 1
        
        source_str = ", ".join([f"{source} ({count})" for source, count in sources.items()])
        print(f"\n{Colors.BOLD}Sources:{Colors.ENDC} {Colors.GREEN}{source_str}{Colors.ENDC}")
        
        # Display timing information
        print(f"{Colors.BOLD}Time:{Colors.ENDC} {total_time:.2f}s (Retrieval: {result['retrieval_time']:.2f}s, Generation: {result['generation_time']:.2f}s)")
        
        # Ask if user wants to see documents
        show_docs = input(f"\nShow retrieved documents? (y/n): ").strip().lower()
        if show_docs == 'y':
            print_section_header("Retrieved Documents")
            for i, (doc, source, score) in enumerate(zip(result['documents'], result['sources'], result['scores'])):
                print_document(i, doc, source, score)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.GREEN}Thank you for using Synthara! Goodbye!{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}An error occurred: {str(e)}{Colors.ENDC}")
        raise
