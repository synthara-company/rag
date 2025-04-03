"""
Simple web interface for the RAG System.
This is a simplified version that doesn't depend on all components.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import List, Dict, Any

# Import simple RAG system
from simple_rag_system import SimpleRAGSystem

# Set page configuration
st.set_page_config(
    page_title="Simple RAG System",
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    # Initialize RAG system
    st.session_state.rag_system = SimpleRAGSystem()

    # Load sample documents
    sample_docs_path = "sample_documents.json"
    if os.path.exists(sample_docs_path):
        st.session_state.rag_system.load_sample_documents(sample_docs_path)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'show_details' not in st.session_state:
    st.session_state.show_details = False

# Sidebar
with st.sidebar:
    st.markdown("## Simple RAG System")

    # Settings section
    with st.expander("Settings", expanded=False):
        top_k = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=5)
        st.checkbox("Show response details", key="show_details")

    # System metrics
    with st.expander("System Metrics", expanded=True):
        metrics = st.session_state.rag_system.get_performance_metrics()

        st.markdown(f"**Total Queries:** {metrics['total_queries']}")
        st.markdown(f"**Documents:** {metrics['document_count']}")

        # Performance metrics
        if metrics['total_queries'] > 0:
            st.markdown("#### Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg. Retrieval Time", f"{metrics['avg_retrieval_time']:.4f}s")
            with col2:
                st.metric("Avg. Generation Time", f"{metrics['avg_generation_time']:.4f}s")

    # Add content section
    with st.expander("Add Content", expanded=False):
        st.markdown("### Add Document")

        # Text input
        doc_text = st.text_area("Enter document text:", height=100)
        doc_source = st.text_input("Source:", "Custom")

        if st.button("Add Document"):
            if doc_text:
                st.session_state.rag_system.add_documents(
                    [doc_text],
                    [doc_source],
                    [{"added_by": "user"}]
                )
                st.success(f"Added document from source '{doc_source}'")
            else:
                st.error("Please enter document text")

# Main content
st.markdown("<h1 class='main-header'>Simple RAG System</h1>", unsafe_allow_html=True)

# Tabs for different sections
tabs = st.tabs(["💬 Chat", "📊 Analytics", "📚 Knowledge Base"])

# Chat Tab
with tabs[0]:
    st.markdown("<h2 class='sub-header'>Interactive Assistant</h2>", unsafe_allow_html=True)

    # Display chat history
    for i, (query, result) in enumerate(st.session_state.chat_history):
        st.markdown(f"<div class='user-message'><b>You:</b> {query}</div>", unsafe_allow_html=True)

        # Display assistant response with source tag
        source_tag = f"<span class='source-tag'>{result['primary_source']}</span>" if 'primary_source' in result else ""
        st.markdown(f"<div class='assistant-message'>{source_tag} <b>Assistant:</b> {result['response']}</div>", unsafe_allow_html=True)

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
            st.rerun()

    if submit_button or (query and query != st.session_state.get("last_query", "")):
        if query:
            st.session_state.last_query = query
            with st.spinner("Generating response..."):
                # Process query
                result = st.session_state.rag_system.process_query(
                    query,
                    top_k=top_k
                )

                # Add to chat history
                st.session_state.chat_history.append((query, result))

                # Rerun to update the UI (this will clear the input field)
                st.rerun()

# Analytics Tab
with tabs[1]:
    st.markdown("<h2 class='sub-header'>Analytics Dashboard</h2>", unsafe_allow_html=True)

    metrics = st.session_state.rag_system.get_performance_metrics()

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", metrics['document_count'])
    with col2:
        st.metric("Total Queries", metrics['total_queries'])
    with col3:
        st.metric("Avg. Total Time", f"{metrics['avg_total_time']:.4f}s")

    # Performance timeline
    if metrics['total_queries'] > 0:
        st.markdown("### Performance Timeline")

        # Create performance dataframe
        perf_data = []
        for i, (retrieval, generation, total) in enumerate(zip(
            st.session_state.rag_system.retrieval_times,
            st.session_state.rag_system.generation_times,
            st.session_state.rag_system.total_times
        )):
            perf_data.append({
                "Query": i + 1,
                "Retrieval Time": retrieval,
                "Generation Time": generation,
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
            retrieved_docs, retrieved_sources, retrieved_metadata, retrieved_scores = st.session_state.rag_system.retrieve(
                search_query,
                top_k=10
            )

            # Create a dataframe for display
            if retrieved_docs:
                results_df = pd.DataFrame({
                    'Source': retrieved_sources,
                    'Document': retrieved_docs,
                    'Relevance': [f"{score:.2f}" for score in retrieved_scores]
                })

                st.markdown(f"Found {len(retrieved_docs)} relevant documents:")
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

# Footer
st.markdown("---")
st.markdown("Simple RAG System - A demonstration of retrieval-augmented generation")

if __name__ == "__main__":
    # This is already running in the Streamlit app
    pass
