"""
Minimal web interface for the RAG System.
This is a very simple version that focuses just on the chat functionality.
"""

import streamlit as st
from simple_rag_system import SimpleRAGSystem

# Set page configuration
st.set_page_config(
    page_title="RAG Chat",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    # Initialize RAG system
    st.session_state.rag_system = SimpleRAGSystem()
    
    # Load sample documents
    st.session_state.rag_system.load_sample_documents("sample_documents.json")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main content
st.title("RAG Chat System")
st.write("Ask questions about machine learning, science fiction movies, or cosmos")

# Simple settings
top_k = st.sidebar.slider("Number of documents to retrieve", min_value=1, max_value=10, value=5)
show_details = st.sidebar.checkbox("Show document details", value=False)

# Chat interface
for i, (query, result) in enumerate(st.session_state.chat_history):
    # User message
    st.write(f"**You:** {query}")
    
    # Assistant message
    st.write(f"**Assistant:** {result['response']}")
    
    # Show document details if requested
    if show_details and 'documents' in result:
        with st.expander("View retrieved documents"):
            for j, (doc, source, score) in enumerate(zip(result['documents'], result['sources'], result['scores'])):
                st.write(f"**Document {j+1}** from {source} (Score: {score:.2f})")
                st.write(doc[:300] + "..." if len(doc) > 300 else doc)
    
    # Add a separator
    st.write("---")

# Query input
query = st.text_input("Enter your question:")

if st.button("Submit") and query:
    with st.spinner("Generating response..."):
        # Process query
        result = st.session_state.rag_system.process_query(
            query,
            top_k=top_k
        )
        
        # Add to chat history
        st.session_state.chat_history.append((query, result))
        
        # Refresh the page to show the new message
        st.rerun()

# Add a clear button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# Add document input in sidebar
st.sidebar.title("Add Document")
doc_text = st.sidebar.text_area("Enter document text:")
doc_source = st.sidebar.text_input("Source:", "Custom")

if st.sidebar.button("Add Document"):
    if doc_text:
        st.session_state.rag_system.add_documents(
            [doc_text],
            [doc_source]
        )
        st.sidebar.success(f"Added document from source '{doc_source}'")
    else:
        st.sidebar.error("Please enter document text")

# Show system info
st.sidebar.title("System Info")
metrics = st.session_state.rag_system.get_performance_metrics()
st.sidebar.write(f"Documents: {metrics['document_count']}")
st.sidebar.write(f"Queries: {metrics['total_queries']}")
if metrics['total_queries'] > 0:
    st.sidebar.write(f"Avg. Retrieval Time: {metrics['avg_retrieval_time']:.4f}s")
    st.sidebar.write(f"Avg. Generation Time: {metrics['avg_generation_time']:.4f}s")
