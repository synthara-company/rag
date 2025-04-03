"""
Advanced web interface for the RAG Transformer system.
"""

import streamlit as st
import matplotlib.pyplot as plt
import os
import sys
import time
import pandas as pd
import numpy as np
from advanced_rag_pipeline import AdvancedRAGTransformer
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="Advanced RAG Transformer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = AdvancedRAGTransformer()
    st.session_state.rag.load_datasets()
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'show_details' not in st.session_state:
    st.session_state.show_details = False
    
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Chat"

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
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Advanced RAG Transformer")
    
    # Settings section
    with st.expander("Settings", expanded=False):
        top_k = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=5)
        st.checkbox("Show response details", key="show_details")
    
    # Knowledge Base section
    with st.expander("Knowledge Base", expanded=True):
        metrics = st.session_state.rag.get_performance_metrics()
        
        st.markdown(f"**Total Documents:** {metrics['knowledge_base_size']}")
        
        # Source distribution chart
        source_data = metrics['source_distribution']
        source_df = pd.DataFrame({
            'Source': list(source_data.keys()),
            'Count': list(source_data.values())
        })
        
        fig = px.bar(source_df, x='Source', y='Count', 
                    title='Knowledge Base Distribution',
                    color='Source', 
                    color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Add content section
    with st.expander("Add Content", expanded=False):
        st.markdown("### Add Web Content")
        web_url = st.text_input("Enter a URL:")
        if st.button("Add Web Content"):
            if web_url:
                with st.spinner(f"Adding content from {web_url}..."):
                    count = st.session_state.rag.add_web_content(web_url)
                    st.success(f"Added {count} document chunks from {web_url}")
            else:
                st.error("Please enter a URL")
        
        st.markdown("### Add Wikipedia Content")
        wiki_topic = st.text_input("Enter a topic:")
        wiki_results = st.slider("Number of articles:", min_value=1, max_value=5, value=2)
        if st.button("Add Wikipedia Content"):
            if wiki_topic:
                with st.spinner(f"Adding Wikipedia content for '{wiki_topic}'..."):
                    count = st.session_state.rag.add_wikipedia_content(wiki_topic, wiki_results)
                    st.success(f"Added {count} Wikipedia articles about {wiki_topic}")
            else:
                st.error("Please enter a topic")
        
        st.markdown("### Add Custom Document")
        custom_doc = st.text_area("Enter a custom document:", height=100)
        custom_source = st.text_input("Source name:", "Custom")
        
        if st.button("Add Document"):
            if custom_doc:
                st.session_state.rag.add_documents([custom_doc], source=custom_source)
                st.success(f"Added document to knowledge base under source '{custom_source}'")
            else:
                st.error("Please enter a document to add")
    
    # Performance metrics section
    with st.expander("Performance Metrics", expanded=False):
        st.markdown(f"**Total Queries:** {metrics['total_queries']}")
        st.markdown(f"**Avg. Retrieval Time:** {metrics['avg_retrieval_time']:.4f} seconds")
        st.markdown(f"**Avg. Generation Time:** {metrics['avg_generation_time']:.4f} seconds")
        st.markdown(f"**Avg. Evaluation Time:** {metrics.get('avg_evaluation_time', 0):.4f} seconds")
        
        if metrics['total_queries'] > 0:
            # Create performance timeline
            if len(st.session_state.rag.retrieval_times) > 0:
                perf_df = pd.DataFrame({
                    'Query': range(1, len(st.session_state.rag.retrieval_times) + 1),
                    'Retrieval Time': st.session_state.rag.retrieval_times,
                    'Generation Time': st.session_state.rag.generation_times,
                    'Evaluation Time': st.session_state.rag.evaluation_times if hasattr(st.session_state.rag, 'evaluation_times') else [0] * len(st.session_state.rag.retrieval_times)
                })
                
                fig = px.line(perf_df, x='Query', y=['Retrieval Time', 'Generation Time', 'Evaluation Time'],
                            title='Performance Timeline',
                            labels={'value': 'Time (seconds)', 'variable': 'Metric'})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

# Main content
st.markdown("<h1 class='main-header'>Advanced RAG Transformer</h1>", unsafe_allow_html=True)

# Tabs for different sections
tabs = st.tabs(["💬 Chat", "📊 Analytics", "📚 Knowledge Base", "ℹ️ About"])

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
                    st.markdown(f"**Evaluation Time:** {result.get('evaluation_time', 0):.4f} seconds")
                
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
    query = st.text_input("Enter your question about machine learning, sci-fi movies, or cosmos:", key="query_input")
    
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
                # Generate response
                start_time = time.time()
                result = st.session_state.rag.generate_response(query, top_k=top_k)
                total_time = time.time() - start_time
                
                # Add to chat history
                st.session_state.chat_history.append((query, result))
                
                # Clear input
                st.session_state.query_input = ""
                
                # Rerun to update the UI
                st.experimental_rerun()

# Analytics Tab
with tabs[1]:
    st.markdown("<h2 class='sub-header'>Analytics Dashboard</h2>", unsafe_allow_html=True)
    
    metrics = st.session_state.rag.get_performance_metrics()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", metrics['knowledge_base_size'])
    with col2:
        st.metric("Total Queries", metrics['total_queries'])
    with col3:
        st.metric("Avg. Retrieval Time", f"{metrics['avg_retrieval_time']:.4f}s")
    with col4:
        st.metric("Avg. Generation Time", f"{metrics['avg_generation_time']:.4f}s")
    
    # Source distribution
    st.markdown("### Knowledge Base Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        source_data = metrics['source_distribution']
        source_df = pd.DataFrame({
            'Source': list(source_data.keys()),
            'Count': list(source_data.values())
        })
        
        fig = px.pie(source_df, values='Count', names='Source', 
                    title='Documents by Source',
                    color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'domain_distribution' in metrics:
            domain_data = metrics['domain_distribution']
            domain_df = pd.DataFrame({
                'Domain': list(domain_data.keys()),
                'Count': list(domain_data.values())
            })
            
            fig = px.pie(domain_df, values='Count', names='Domain', 
                        title='Documents by Domain',
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
    
    # Conversation history
    if metrics['total_queries'] > 0:
        st.markdown("### Conversation History")
        
        history_df = st.session_state.rag.get_conversation_history()
        
        # Add source information
        sources = []
        for _, (query, result) in enumerate(st.session_state.chat_history):
            if 'primary_source' in result:
                sources.append(result['primary_source'])
            else:
                sources.append("Unknown")
        
        if len(sources) == len(history_df):
            history_df['primary_source'] = sources
        
        st.dataframe(history_df, use_container_width=True)
        
        # Performance timeline
        st.markdown("### Performance Timeline")
        
        if len(st.session_state.rag.retrieval_times) > 0:
            perf_df = pd.DataFrame({
                'Query': range(1, len(st.session_state.rag.retrieval_times) + 1),
                'Retrieval Time': st.session_state.rag.retrieval_times,
                'Generation Time': st.session_state.rag.generation_times,
                'Evaluation Time': st.session_state.rag.evaluation_times if hasattr(st.session_state.rag, 'evaluation_times') else [0] * len(st.session_state.rag.retrieval_times)
            })
            
            fig = px.line(perf_df.melt(id_vars=['Query'], var_name='Metric', value_name='Time'), 
                        x='Query', y='Time', color='Metric',
                        title='Performance Metrics Over Time',
                        labels={'Time': 'Time (seconds)'})
            st.plotly_chart(fig, use_container_width=True)

# Knowledge Base Tab
with tabs[2]:
    st.markdown("<h2 class='sub-header'>Knowledge Base Explorer</h2>", unsafe_allow_html=True)
    
    # Search functionality
    st.markdown("### Search Documents")
    search_query = st.text_input("Enter search terms:")
    
    if search_query:
        with st.spinner("Searching..."):
            # Use the RAG retrieval to find relevant documents
            docs, sources, metadata, scores = st.session_state.rag.retrieve_context(search_query, top_k=10)
            
            # Create a dataframe for display
            results_df = pd.DataFrame({
                'Source': sources,
                'Document': docs,
                'Relevance': [f"{score:.2f}" for score in scores]
            })
            
            st.markdown(f"Found {len(docs)} relevant documents:")
            st.dataframe(results_df, use_container_width=True)
    
    # Document statistics
    st.markdown("### Document Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Document length distribution
        doc_lengths = [len(doc.split()) for doc in st.session_state.rag.knowledge_base]
        
        fig = px.histogram(
            x=doc_lengths, 
            nbins=20,
            labels={'x': 'Document Length (words)', 'y': 'Count'},
            title='Document Length Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Source distribution
        source_data = st.session_state.rag.get_performance_metrics()['source_distribution']
        source_df = pd.DataFrame({
            'Source': list(source_data.keys()),
            'Count': list(source_data.values())
        }).sort_values('Count', ascending=False)
        
        fig = px.bar(
            source_df, 
            x='Source', 
            y='Count',
            title='Documents by Source',
            color='Source'
        )
        st.plotly_chart(fig, use_container_width=True)

# About Tab
with tabs[3]:
    st.markdown("<h2 class='sub-header'>About RAG Transformer</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### What is RAG Transformer?
    
    RAG (Retrieval-Augmented Generation) Transformer is an intelligent AI assistant that combines advanced machine learning techniques to provide contextually rich, informative responses across multiple knowledge domains.
    
    ### Key Capabilities:
    - **Intelligent Knowledge Retrieval**: Searches through a diverse dataset of machine learning, science fiction, and cosmic content
    - **Context-Aware Responses**: Generates answers that are not just accurate, but contextually relevant
    - **Multidomain Intelligence**: Bridges insights from:
      1. Machine Learning Concepts
      2. Science Fiction Movies
      3. Cosmic and Astronomical Observations
    
    ### How It Works
    
    The system uses a sophisticated two-step process:
    1. **Retrieval**: Finds the most relevant documents for a given query
    2. **Generation**: Creates a coherent, informative response using the retrieved context
    
    ### Technical Architecture
    - **Embedding Model**: `all-MiniLM-L6-v2`
    - **Generation Model**: `google/flan-t5-small`
    - **Indexing**: FAISS semantic search
    - **Data Sources**: TMDB and NASA APIs, Wikipedia, Web content
    
    ### Advanced Features
    - Interactive query interface
    - Adaptive learning
    - Diverse knowledge base
    - Real-time response generation
    - Performance analytics
    - Knowledge base expansion
    """)
    
    # Architecture diagram
    st.markdown("### System Architecture")
    
    # Create a simple architecture diagram
    from graphviz import Digraph
    
    graph = Digraph()
    graph.attr(rankdir='TB', size='8,5')
    
    # Add nodes
    graph.node('User', shape='ellipse')
    graph.node('Query', shape='box')
    graph.node('RAG', shape='box', style='filled', fillcolor='lightblue')
    graph.node('Retriever', shape='box', style='filled', fillcolor='lightgreen')
    graph.node('Generator', shape='box', style='filled', fillcolor='lightgreen')
    graph.node('KB', label='Knowledge Base', shape='cylinder', style='filled', fillcolor='lightyellow')
    graph.node('Response', shape='box')
    
    # Add edges
    graph.edge('User', 'Query')
    graph.edge('Query', 'RAG')
    graph.edge('RAG', 'Retriever')
    graph.edge('Retriever', 'KB', dir='both')
    graph.edge('Retriever', 'Generator')
    graph.edge('Generator', 'Response')
    graph.edge('Response', 'User')
    
    # Render the graph
    st.graphviz_chart(graph)

# Footer
st.markdown("---")
st.markdown("Advanced RAG Transformer combines machine learning, sci-fi, and cosmos knowledge to provide informative responses.")
st.markdown("Built with Streamlit, Hugging Face Transformers, and FAISS.")

if __name__ == "__main__":
    # This is already running in the Streamlit app
    pass
