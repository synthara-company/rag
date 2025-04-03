"""
Web interface for the Unified RAG System.
This module provides a Streamlit-based web interface for the Unified RAG System.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
from datetime import datetime
import uuid
from typing import List, Dict, Any, Tuple, Optional, Union

# Import unified RAG system
from unified_rag_system import UnifiedRAGSystem

# Set page configuration
st.set_page_config(
    page_title="Unified RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F0F0;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4B8BBE;
        color: white;
    }
    .image-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .image-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
        max-width: 200px;
    }
    .image-caption {
        font-size: 0.8rem;
        color: #666;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    # Load configuration
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Get configuration from config or use defaults
    use_hybrid_retrieval = config.get('use_hybrid_retrieval', True)
    use_multimodal = config.get('use_multimodal', True)
    use_personalization = config.get('use_personalization', True)
    use_vector_db = config.get('use_vector_db', True)
    vector_db_type = config.get('vector_db_type', 'chroma')
    
    # Initialize RAG system
    st.session_state.rag_system = UnifiedRAGSystem(
        config_path=config_path,
        use_hybrid_retrieval=use_hybrid_retrieval,
        use_multimodal=use_multimodal,
        use_personalization=use_personalization,
        use_vector_db=use_vector_db,
        vector_db_type=vector_db_type
    )
    
    # Load sample documents if available
    sample_docs_path = "sample_documents.json"
    if os.path.exists(sample_docs_path):
        try:
            with open(sample_docs_path, 'r') as f:
                sample_docs = json.load(f)
            
            documents = sample_docs.get('documents', [])
            sources = sample_docs.get('sources', [])
            metadata = sample_docs.get('metadata', [])
            
            if documents:
                st.session_state.rag_system.add_documents(documents, sources, metadata)
                print(f"Loaded {len(documents)} sample documents")
        except Exception as e:
            print(f"Error loading sample documents: {e}")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_user_id' not in st.session_state:
    st.session_state.current_user_id = str(uuid.uuid4())

if 'show_details' not in st.session_state:
    st.session_state.show_details = False

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Chat"

# Sidebar
with st.sidebar:
    st.markdown("## Unified RAG System")
    
    # User selection
    st.markdown("### User")
    user_id = st.text_input("User ID:", value=st.session_state.current_user_id)
    if user_id != st.session_state.current_user_id:
        st.session_state.current_user_id = user_id
    
    # Settings section
    with st.expander("Settings", expanded=False):
        text_top_k = st.slider("Number of text documents to retrieve", min_value=1, max_value=10, value=5)
        image_top_k = st.slider("Number of images to retrieve", min_value=1, max_value=10, value=3)
        include_images = st.checkbox("Include images in response", value=True)
        st.checkbox("Show response details", key="show_details")
        
        # Component toggles
        st.markdown("### Components")
        use_hybrid_retrieval = st.checkbox("Use hybrid retrieval", value=st.session_state.rag_system.use_hybrid_retrieval)
        use_multimodal = st.checkbox("Use multi-modal capabilities", value=st.session_state.rag_system.use_multimodal)
        use_personalization = st.checkbox("Use personalization", value=st.session_state.rag_system.use_personalization)
        use_vector_db = st.checkbox("Use vector database", value=st.session_state.rag_system.use_vector_db)
        
        # Apply settings
        if st.button("Apply Settings"):
            # Reinitialize RAG system with new settings
            st.session_state.rag_system = UnifiedRAGSystem(
                use_hybrid_retrieval=use_hybrid_retrieval,
                use_multimodal=use_multimodal,
                use_personalization=use_personalization,
                use_vector_db=use_vector_db,
                vector_db_type='chroma'  # Default to ChromaDB for simplicity
            )
            
            st.success("Settings applied")
    
    # System metrics
    with st.expander("System Metrics", expanded=True):
        metrics = st.session_state.rag_system.get_performance_metrics()
        
        st.markdown(f"**Total Queries:** {metrics['total_queries']}")
        st.markdown(f"**Documents:** {metrics['document_count']}")
        st.markdown(f"**Images:** {metrics['image_count']}")
        
        # Performance metrics
        if metrics['total_queries'] > 0:
            st.markdown("#### Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg. Retrieval Time", f"{metrics['avg_retrieval_time']:.4f}s")
                st.metric("Avg. Generation Time", f"{metrics['avg_generation_time']:.4f}s")
            with col2:
                st.metric("Avg. Evaluation Time", f"{metrics['avg_evaluation_time']:.4f}s")
                st.metric("Avg. Total Time", f"{metrics['avg_total_time']:.4f}s")
    
    # User profile
    if st.session_state.rag_system.use_personalization:
        with st.expander("User Profile", expanded=False):
            user_stats = st.session_state.rag_system.get_user_stats(st.session_state.current_user_id)
            
            if user_stats:
                st.markdown(f"**User:** {user_stats['name']}")
                st.markdown(f"**Queries:** {user_stats['total_queries']}")
                st.markdown(f"**Interactions:** {user_stats['total_interactions']}")
                
                # Interests
                if user_stats['interests']:
                    st.markdown("#### Interests")
                    for interest, weight in user_stats['interests']:
                        st.progress(weight, text=interest)
                
                # Frequent terms
                if user_stats['frequent_terms']:
                    st.markdown("#### Frequent Terms")
                    terms_df = pd.DataFrame(
                        user_stats['frequent_terms'],
                        columns=['Term', 'Count']
                    )
                    fig = px.bar(terms_df, x='Term', y='Count')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("No user profile data available")
    
    # Add content section
    with st.expander("Add Content", expanded=False):
        st.markdown("### Add Documents")
        
        # Text input
        doc_text = st.text_area("Enter document text:", height=100)
        doc_source = st.text_input("Source:", "Custom")
        
        if st.button("Add Document"):
            if doc_text:
                st.session_state.rag_system.add_documents(
                    [doc_text],
                    [doc_source],
                    [{"added_by": st.session_state.current_user_id, "timestamp": datetime.now().isoformat()}]
                )
                st.success(f"Added document from source '{doc_source}'")
            else:
                st.error("Please enter document text")
        
        # File upload
        st.markdown("### Upload Document")
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'md'])
        upload_source = st.text_input("Upload source:", "Uploaded")
        
        if uploaded_file is not None and st.button("Upload"):
            try:
                # For simplicity, we only handle text files here
                # In a real application, you would want to handle PDFs and other formats
                content = uploaded_file.read().decode()
                
                st.session_state.rag_system.add_documents(
                    [content],
                    [upload_source],
                    [{"filename": uploaded_file.name, "added_by": st.session_state.current_user_id, "timestamp": datetime.now().isoformat()}]
                )
                
                st.success(f"Uploaded {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error uploading file: {e}")
        
        # Image upload (if multi-modal is enabled)
        if st.session_state.rag_system.use_multimodal:
            st.markdown("### Upload Image")
            uploaded_image = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
            image_caption = st.text_input("Image caption:", "")
            image_source = st.text_input("Image source:", "Uploaded")
            
            if uploaded_image is not None and st.button("Upload Image"):
                try:
                    image = Image.open(uploaded_image)
                    
                    st.session_state.rag_system.add_images(
                        [image],
                        [image_caption] if image_caption else None,
                        [image_source],
                        [{"filename": uploaded_image.name, "added_by": st.session_state.current_user_id, "timestamp": datetime.now().isoformat()}]
                    )
                    
                    st.success(f"Uploaded image {uploaded_image.name}")
                except Exception as e:
                    st.error(f"Error uploading image: {e}")

# Main content
st.markdown("<h1 class='main-header'>Unified RAG System</h1>", unsafe_allow_html=True)

# Tabs for different sections
tabs = st.tabs(["💬 Chat", "📊 Analytics", "📚 Knowledge Base", "🖼️ Images", "ℹ️ About"])

# Chat Tab
with tabs[0]:
    st.markdown("<h2 class='sub-header'>Interactive Assistant</h2>", unsafe_allow_html=True)
    
    # Display chat history
    for i, (query, result) in enumerate(st.session_state.chat_history):
        st.markdown(f"<div class='user-message'><b>You:</b> {query}</div>", unsafe_allow_html=True)
        
        # Display assistant response with source tag
        source_tag = f"<span class='source-tag'>{result['primary_source']}</span>" if 'primary_source' in result else ""
        st.markdown(f"<div class='assistant-message'>{source_tag} <b>Assistant:</b> {result['response']}</div>", unsafe_allow_html=True)
        
        # Display images if available
        if 'images' in result and result['images']:
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            for j, (img_data, caption) in enumerate(zip(result['images'], result['image_captions'])):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(f"data:image/jpeg;base64,{img_data}", caption=f"Image {j+1}", width=150)
                with col2:
                    st.markdown(f"**Caption:** {caption}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display metrics in a collapsible section
        if st.session_state.show_details:
            with st.expander("View details", expanded=False):
                st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Retrieval Time:** {result.get('retrieval_time', 0):.4f} seconds")
                with col2:
                    st.markdown(f"**Generation Time:** {result.get('generation_time', 0):.4f} seconds")
                with col3:
                    st.markdown(f"**Total Time:** {result.get('total_time', 0):.4f} seconds")
                
                # Display sources and context
                if 'documents' in result and result['documents']:
                    st.markdown("#### Retrieved Documents")
                    for j, (doc, source, score) in enumerate(zip(result['documents'], result['sources'], result['scores'])):
                        st.markdown(f"**Document {j+1}** [{source}] (Score: {score:.2f})")
                        st.markdown(f"{doc[:300]}..." if len(doc) > 300 else doc)
                
                # Display evaluation if available
                if 'evaluation' in result:
                    st.markdown("#### Response Evaluation")
                    eval_result = result['evaluation']
                    st.markdown(f"**Quality:** {eval_result.get('quality', 'Unknown')}")
                    if 'issue' in eval_result:
                        st.markdown(f"**Issue:** {eval_result['issue']}")
                    if 'suggestion' in eval_result:
                        st.markdown(f"**Suggestion:** {eval_result['suggestion']}")
                    if 'query_relevance' in eval_result:
                        st.markdown(f"**Query Relevance:** {eval_result['query_relevance']:.2f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Query input
    st.markdown("### Ask a question")
    query = st.text_input("Enter your question:", key="query_input")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Submit")
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.experimental_rerun()
    
    if submit_button or (query and query != st.session_state.get("last_query", "")):
        if query:
            st.session_state.last_query = query
            with st.spinner("Generating response..."):
                # Process query
                result = st.session_state.rag_system.process_query(
                    query,
                    user_id=st.session_state.current_user_id,
                    include_images=include_images,
                    text_top_k=text_top_k,
                    image_top_k=image_top_k
                )
                
                # Add to chat history
                st.session_state.chat_history.append((query, result))
                
                # Clear input
                st.session_state.query_input = ""
                
                # Log interaction
                if st.session_state.rag_system.use_personalization:
                    st.session_state.rag_system.log_user_interaction(
                        st.session_state.current_user_id,
                        "query",
                        query,
                        {"response_id": result.get("timestamp", "")}
                    )
                
                # Rerun to update the UI
                st.experimental_rerun()

# Analytics Tab
with tabs[1]:
    st.markdown("<h2 class='sub-header'>Analytics Dashboard</h2>", unsafe_allow_html=True)
    
    metrics = st.session_state.rag_system.get_performance_metrics()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", metrics['document_count'])
    with col2:
        st.metric("Total Images", metrics['image_count'])
    with col3:
        st.metric("Total Queries", metrics['total_queries'])
    with col4:
        st.metric("Avg. Total Time", f"{metrics['avg_total_time']:.4f}s")
    
    # Performance timeline
    if metrics['total_queries'] > 0:
        st.markdown("### Performance Timeline")
        
        # Create performance dataframe
        perf_data = []
        for i, (retrieval, generation, evaluation, total) in enumerate(zip(
            st.session_state.rag_system.retrieval_times,
            st.session_state.rag_system.generation_times,
            st.session_state.rag_system.evaluation_times,
            st.session_state.rag_system.total_times
        )):
            perf_data.append({
                "Query": i + 1,
                "Retrieval Time": retrieval,
                "Generation Time": generation,
                "Evaluation Time": evaluation,
                "Total Time": total
            })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            
            # Plot performance timeline
            fig = px.line(
                perf_df.melt(id_vars=['Query'], var_name='Metric', value_name='Time'),
                x='Query',
                y='Time',
                color='Metric',
                title='Performance Metrics Over Time',
                labels={'Time': 'Time (seconds)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Query history
    if st.session_state.chat_history:
        st.markdown("### Query History")
        
        # Create query history dataframe
        history_data = []
        for i, (query, result) in enumerate(st.session_state.chat_history):
            history_data.append({
                "Query": query,
                "Response Length": len(result['response']),
                "Documents": len(result['documents']),
                "Images": len(result.get('images', [])),
                "Primary Source": result['primary_source'],
                "Total Time": result['total_time']
            })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
            
            # Plot query distribution by source
            source_counts = history_df['Primary Source'].value_counts().reset_index()
            source_counts.columns = ['Source', 'Count']
            
            fig = px.pie(
                source_counts,
                values='Count',
                names='Source',
                title='Queries by Primary Source'
            )
            st.plotly_chart(fig, use_container_width=True)

# Knowledge Base Tab
with tabs[2]:
    st.markdown("<h2 class='sub-header'>Knowledge Base Explorer</h2>", unsafe_allow_html=True)
    
    # Search functionality
    st.markdown("### Search Documents")
    search_query = st.text_input("Enter search terms:")
    
    if search_query:
        with st.spinner("Searching..."):
            # Use the RAG system to find relevant documents
            result = st.session_state.rag_system.process_query(
                search_query,
                include_images=False,
                text_top_k=10
            )
            
            # Create a dataframe for display
            if 'documents' in result and result['documents']:
                results_df = pd.DataFrame({
                    'Source': result['sources'],
                    'Document': result['documents'],
                    'Relevance': [f"{score:.2f}" for score in result['scores']]
                })
                
                st.markdown(f"Found {len(result['documents'])} relevant documents:")
                st.dataframe(results_df, use_container_width=True)
            else:
                st.markdown("No relevant documents found")
    
    # Document statistics
    if st.session_state.rag_system.documents:
        st.markdown("### Document Statistics")
        
        # Create document statistics
        doc_stats = []
        for doc, source in zip(st.session_state.rag_system.documents, st.session_state.rag_system.document_sources):
            doc_stats.append({
                "Source": source,
                "Length": len(doc.split())
            })
        
        doc_stats_df = pd.DataFrame(doc_stats)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Document length distribution
            fig = px.histogram(
                doc_stats_df,
                x='Length',
                nbins=20,
                labels={'Length': 'Document Length (words)'},
                title='Document Length Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Source distribution
            source_counts = doc_stats_df['Source'].value_counts().reset_index()
            source_counts.columns = ['Source', 'Count']
            
            fig = px.bar(
                source_counts,
                x='Source',
                y='Count',
                title='Documents by Source',
                color='Source'
            )
            st.plotly_chart(fig, use_container_width=True)

# Images Tab
with tabs[3]:
    st.markdown("<h2 class='sub-header'>Image Gallery</h2>", unsafe_allow_html=True)
    
    # Display images if available
    if st.session_state.rag_system.use_multimodal and hasattr(st.session_state.rag_system, 'multimodal_rag'):
        if hasattr(st.session_state.rag_system.multimodal_rag, 'images') and st.session_state.rag_system.multimodal_rag.images:
            # Image search
            st.markdown("### Search Images")
            image_query = st.text_input("Enter image search terms:")
            
            if image_query:
                with st.spinner("Searching images..."):
                    # Use the multi-modal RAG to find relevant images
                    images, captions, sources, metadata, scores = st.session_state.rag_system.multimodal_rag.retrieve_images(
                        image_query,
                        top_k=9
                    )
                    
                    if images:
                        st.markdown(f"Found {len(images)} relevant images:")
                        
                        # Display images in a grid
                        cols = st.columns(3)
                        for i, (img, caption, source, score) in enumerate(zip(images, captions, sources, scores)):
                            with cols[i % 3]:
                                st.image(img, caption=f"{caption} [{source}] (Score: {score:.2f})")
                    else:
                        st.markdown("No relevant images found")
            
            # Display all images
            st.markdown("### All Images")
            
            # Get all images
            images = st.session_state.rag_system.multimodal_rag.images
            captions = st.session_state.rag_system.multimodal_rag.image_captions
            sources = st.session_state.rag_system.multimodal_rag.image_sources
            
            # Display images in a grid
            cols = st.columns(3)
            for i, (img, caption, source) in enumerate(zip(images, captions, sources)):
                with cols[i % 3]:
                    st.image(img, caption=f"{caption} [{source}]")
        else:
            st.markdown("No images available in the system")
    else:
        st.markdown("Multi-modal capabilities are not enabled")

# About Tab
with tabs[4]:
    st.markdown("<h2 class='sub-header'>About Unified RAG System</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### What is the Unified RAG System?
    
    The Unified RAG System is a comprehensive Retrieval-Augmented Generation system that combines multiple advanced components:
    
    1. **Hybrid Retrieval**: Combines dense and sparse retrieval methods for better document retrieval
    2. **Multi-Modal Capabilities**: Handles both text and image inputs and queries
    3. **Personalization**: Adapts responses based on user preferences and history
    4. **Vector Database Integration**: Uses vector databases for efficient storage and retrieval
    
    ### Key Features
    
    - **Advanced Prompting**: Uses few-shot examples and domain-specific templates
    - **Response Evaluation**: Assesses and improves the quality of generated responses
    - **Multiple Model Support**: Integrates with Hugging Face, OpenAI, and Anthropic models
    - **Knowledge Graph Visualization**: Creates visual representations of relationships between concepts
    - **Web Interface**: User-friendly interface with analytics and knowledge base exploration
    
    ### Architecture
    
    The system consists of several components:
    
    - **Retriever**: Finds the most relevant documents for a given query
    - **Generator**: Creates coherent, informative responses using the retrieved context
    - **Evaluator**: Assesses and improves the quality of generated responses
    - **Personalizer**: Adapts responses based on user preferences and history
    - **Vector Database**: Stores and retrieves document embeddings efficiently
    
    ### Technologies Used
    
    - **Python**: Core programming language
    - **Hugging Face Transformers**: For language models and embeddings
    - **FAISS**: For efficient similarity search
    - **Streamlit**: For the web interface
    - **Plotly**: For interactive visualizations
    - **ChromaDB/Pinecone/Qdrant**: For vector database storage
    """)
    
    # Component status
    st.markdown("### Component Status")
    
    components = {
        "Hybrid Retrieval": st.session_state.rag_system.use_hybrid_retrieval,
        "Multi-Modal": st.session_state.rag_system.use_multimodal,
        "Personalization": st.session_state.rag_system.use_personalization,
        "Vector Database": st.session_state.rag_system.use_vector_db
    }
    
    for component, status in components.items():
        st.markdown(f"- **{component}**: {'✅ Enabled' if status else '❌ Disabled'}")

# Footer
st.markdown("---")
st.markdown("Unified RAG System - Combining hybrid retrieval, multi-modal capabilities, personalization, and vector database integration")

if __name__ == "__main__":
    # This is already running in the Streamlit app
    pass
