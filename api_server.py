"""
API server for the RAG Transformer system.
This module provides a FastAPI-based API for the RAG Transformer.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Query, Body, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
from dotenv import load_dotenv
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Import RAG components
from advanced_rag_pipeline import AdvancedRAGTransformer
from knowledge_graph_visualizer import KnowledgeGraphBuilder, KnowledgeGraphVisualizer
from model_integrations import ModelFactory, BaseLanguageModel

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Transformer API",
    description="API for the RAG Transformer system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Transformer
rag_transformer = AdvancedRAGTransformer()
rag_transformer.load_datasets()

# Initialize knowledge graph builder
kg_builder = KnowledgeGraphBuilder(use_spacy=True, embedding_model="all-MiniLM-L6-v2")

# Initialize model factory
model_factory = ModelFactory()

# Define API models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    top_k: int = Field(5, description="Number of documents to retrieve")

class QueryResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    sources: List[str] = Field([], description="Sources of information")
    primary_source: str = Field("Unknown", description="Primary source of information")
    retrieval_time: float = Field(0.0, description="Time taken for retrieval")
    generation_time: float = Field(0.0, description="Time taken for generation")
    evaluation_time: float = Field(0.0, description="Time taken for evaluation")
    context_docs: List[str] = Field([], description="Retrieved context documents")
    similarity_scores: List[float] = Field([], description="Similarity scores for retrieved documents")

class DocumentRequest(BaseModel):
    content: str = Field(..., description="Document content")
    source: str = Field("Custom", description="Document source")
    metadata: Dict[str, Any] = Field({}, description="Document metadata")

class WebContentRequest(BaseModel):
    url: str = Field(..., description="URL to fetch")
    source_name: Optional[str] = Field(None, description="Source name")

class WikipediaRequest(BaseModel):
    topic: str = Field(..., description="Topic to search for")
    max_results: int = Field(3, description="Maximum number of results")

class KnowledgeGraphRequest(BaseModel):
    min_edge_weight: float = Field(0.5, description="Minimum edge weight")
    min_node_frequency: int = Field(2, description="Minimum node frequency")
    max_nodes: int = Field(100, description="Maximum number of nodes")
    format: str = Field("json", description="Output format (json, png, svg)")

class ModelGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    model_type: str = Field("huggingface", description="Model type (huggingface, openai, anthropic)")
    model_name: Optional[str] = Field(None, description="Model name")
    max_tokens: int = Field(150, description="Maximum number of tokens to generate")

# API endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RAG Transformer API is running"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Generate a response for a query
    """
    try:
        result = rag_transformer.generate_response(request.query, top_k=request.top_k)
        
        return {
            "response": result["response"],
            "sources": result.get("sources", []),
            "primary_source": result.get("primary_source", "Unknown"),
            "retrieval_time": result.get("retrieval_time", 0.0),
            "generation_time": result.get("generation_time", 0.0),
            "evaluation_time": result.get("evaluation_time", 0.0),
            "context_docs": result.get("context_docs", []),
            "similarity_scores": result.get("similarity_scores", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/documents/add")
async def add_document(request: DocumentRequest):
    """
    Add a document to the knowledge base
    """
    try:
        rag_transformer.add_documents([{
            "content": request.content,
            "metadata": request.metadata
        }], source=request.source)
        
        return {"message": f"Document added to knowledge base under source '{request.source}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

@app.post("/documents/add_web")
async def add_web_content(request: WebContentRequest):
    """
    Add content from a web page to the knowledge base
    """
    try:
        count = rag_transformer.add_web_content(request.url, request.source_name)
        
        return {"message": f"Added {count} document chunks from {request.url}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding web content: {str(e)}")

@app.post("/documents/add_wikipedia")
async def add_wikipedia_content(request: WikipediaRequest):
    """
    Add Wikipedia content on a topic to the knowledge base
    """
    try:
        count = rag_transformer.add_wikipedia_content(request.topic, request.max_results)
        
        return {"message": f"Added {count} Wikipedia articles about {request.topic}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding Wikipedia content: {str(e)}")

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    source: str = Query("Uploaded"),
):
    """
    Upload a document file to the knowledge base
    """
    try:
        content = await file.read()
        text_content = content.decode("utf-8")
        
        rag_transformer.add_documents([text_content], source=source)
        
        return {"message": f"Document '{file.filename}' uploaded and added to knowledge base under source '{source}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/knowledge_base/stats")
async def get_knowledge_base_stats():
    """
    Get statistics about the knowledge base
    """
    try:
        metrics = rag_transformer.get_performance_metrics()
        
        return {
            "knowledge_base_size": metrics["knowledge_base_size"],
            "source_distribution": metrics["source_distribution"],
            "domain_distribution": metrics.get("domain_distribution", {}),
            "total_queries": metrics["total_queries"],
            "avg_retrieval_time": metrics["avg_retrieval_time"],
            "avg_generation_time": metrics["avg_generation_time"],
            "avg_evaluation_time": metrics.get("avg_evaluation_time", 0.0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting knowledge base stats: {str(e)}")

@app.post("/knowledge_graph/generate")
async def generate_knowledge_graph(request: KnowledgeGraphRequest):
    """
    Generate a knowledge graph from the knowledge base
    """
    try:
        # Build knowledge graph
        graph = kg_builder.build_graph_from_documents(
            rag_transformer.knowledge_base,
            rag_transformer.document_sources
        )
        
        # Prune graph
        pruned_graph = kg_builder.prune_graph(
            min_edge_weight=request.min_edge_weight,
            min_node_frequency=request.min_node_frequency,
            max_nodes=request.max_nodes
        )
        
        # Generate output based on requested format
        if request.format == "json":
            # Export graph to JSON
            graph_data = KnowledgeGraphVisualizer.export_graph_json(pruned_graph, "knowledge_graph.json")
            return graph_data
        
        elif request.format in ["png", "svg"]:
            # Visualize graph
            fig = KnowledgeGraphVisualizer.visualize_matplotlib(
                pruned_graph,
                "knowledge_graph.png",
                "Knowledge Graph"
            )
            
            # Convert figure to image
            buf = io.BytesIO()
            fig.savefig(buf, format=request.format)
            buf.seek(0)
            
            # Return image
            return FileResponse(
                buf,
                media_type=f"image/{request.format}",
                headers={"Content-Disposition": f"attachment; filename=knowledge_graph.{request.format}"}
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating knowledge graph: {str(e)}")

@app.post("/models/generate")
async def generate_with_model(request: ModelGenerationRequest):
    """
    Generate text using a specified model
    """
    try:
        # Create model
        model = model_factory.create_model(request.model_type, request.model_name)
        
        # Generate text
        result = model.generate(request.prompt, max_tokens=request.max_tokens)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

@app.get("/models/info")
async def get_model_info(
    model_type: str = Query("huggingface", description="Model type"),
    model_name: Optional[str] = Query(None, description="Model name")
):
    """
    Get information about a model
    """
    try:
        # Create model
        model = model_factory.create_model(model_type, model_name)
        
        # Get model info
        info = model.get_model_info()
        
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.get("/conversation/history")
async def get_conversation_history():
    """
    Get conversation history
    """
    try:
        history_df = rag_transformer.get_conversation_history()
        
        # Convert DataFrame to list of dictionaries
        history = history_df.to_dict(orient="records")
        
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting conversation history: {str(e)}")

def start():
    """Start the API server"""
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()
