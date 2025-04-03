import streamlit as st
import matplotlib.pyplot as plt
import os
import sys
import time
import pandas as pd
import numpy as np
from enhanced_rag_pipeline import EnhancedRAGTransformer

# Set page configuration
st.set_page_config(
    page_title="RAG Transformer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = EnhancedRAGTransformer()
    st.session_state.rag.load_datasets()
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #306998;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E8F4F8;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #F0F0F0;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .source-tag {
        background-color: #4B8BBE;
        color: white;
        padding: 2px 5px;
        border-radius: 5px;
        font-size: 0.8rem;
        margin-right: 5px;
    }
    .metrics-container {
        background-color: #F8F9FA;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## RAG Transformer")
    st.markdown("### Settings")
    
    top_k = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=5)
    
    st.markdown("### Knowledge Base")
    metrics = st.session_state.rag.get_performance_metrics()
    
    st.markdown("#### Source Distribution")
    source_data = metrics['source_distribution']
    source_df = pd.DataFrame({
        'Source': list(source_data.keys()),
        'Count': list(source_data.values())
    })
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(source_df['Source'], source_df['Count'], color=['#4B8BBE', '#FFD43B', '#306998'])
    ax.set_ylabel('Number of Documents')
    ax.set_title('Knowledge Base Distribution')
    st.pyplot(fig)
    
    st.markdown("#### Performance Metrics")
    st.markdown(f"**Total Queries:** {metrics['total_queries']}")
    st.markdown(f"**Knowledge Base Size:** {metrics['knowledge_base_size']}")
    st.markdown(f"**Avg. Retrieval Time:** {metrics['avg_retrieval_time']:.4f} seconds")
    st.markdown(f"**Avg. Generation Time:** {metrics['avg_generation_time']:.4f} seconds")
    
    # Add custom documents
    st.markdown("### Add Custom Documents")
    custom_doc = st.text_area("Enter a custom document:", height=100)
    custom_source = st.text_input("Source name:", "Custom")
    
    if st.button("Add Document"):
        if custom_doc:
            st.session_state.rag.add_documents([custom_doc], source=custom_source)
            st.success(f"Added document to knowledge base under source '{custom_source}'")
        else:
            st.error("Please enter a document to add")

# Main content
st.markdown("<h1 class='main-header'>RAG Transformer</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>Machine Learning, Sci-Fi, and Cosmos Assistant</h2>", unsafe_allow_html=True)

# Display chat history
for i, (query, result) in enumerate(st.session_state.chat_history):
    st.markdown(f"<div class='user-message'><b>You:</b> {query}</div>", unsafe_allow_html=True)
    
    # Display assistant response with source tag
    source_tag = f"<span class='source-tag'>{result['primary_source']}</span>" if 'primary_source' in result else ""
    st.markdown(f"<div class='assistant-message'>{source_tag} <b>Assistant:</b> {result['response']}</div>", unsafe_allow_html=True)
    
    # Display metrics in a collapsible section
    with st.expander("View details", expanded=False):
        st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
        
        # Display metrics
        if 'retrieval_time' in result and 'generation_time' in result:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Retrieval Time:** {result['retrieval_time']:.4f} seconds")
            with col2:
                st.markdown(f"**Generation Time:** {result['generation_time']:.4f} seconds")
        
        # Display sources and context
        if 'sources' in result and 'context_docs' in result:
            st.markdown("#### Sources")
            sources = result.get('sources', [])
            if sources:
                source_counts = {}
                for source in sources:
                    if source in source_counts:
                        source_counts[source] += 1
                    else:
                        source_counts[source] = 1
                
                for source, count in source_counts.items():
                    st.markdown(f"- {source}: {count} documents")
            
            st.markdown("#### Retrieved Context")
            context_docs = result.get('context_docs', [])
            if context_docs:
                for i, doc in enumerate(context_docs[:3]):  # Show only top 3 for brevity
                    st.markdown(f"**Document {i+1}:** {doc[:200]}..." if len(doc) > 200 else f"**Document {i+1}:** {doc}")
                
                if len(context_docs) > 3:
                    st.markdown(f"*...and {len(context_docs) - 3} more documents*")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Query input
query = st.text_input("Ask a question about machine learning, sci-fi movies, or cosmos:")

if st.button("Submit") or query:
    if query:
        with st.spinner("Generating response..."):
            # Generate response
            start_time = time.time()
            result = st.session_state.rag.generate_response(query)
            total_time = time.time() - start_time
            
            # Add to chat history
            st.session_state.chat_history.append((query, result))
            
            # Clear input
            query = ""
            
            # Rerun to update the UI
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("RAG Transformer combines machine learning, sci-fi, and cosmos knowledge to provide informative responses.")
st.markdown("Built with Streamlit, Hugging Face Transformers, and FAISS.")

if __name__ == "__main__":
    # This is already running in the Streamlit app
    pass
