"""
Document loader for the RAG Transformer system.
This module provides functionality to load documents from various sources.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional, Union
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import urlparse

class DocumentLoader:
    """Class for loading documents from various sources"""
    
    def __init__(self, base_dir: str = os.path.join(os.path.dirname(__file__), 'datasets')):
        """
        Initialize document loader
        
        Args:
            base_dir (str): Base directory for storing datasets
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def load_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load documents from a JSON file
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            List[Dict[str, Any]]: List of documents
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        except Exception as e:
            print(f"Error loading JSON file {file_path}: {e}")
            return []
    
    def load_text_file(self, file_path: str, chunk_size: int = 1000) -> List[str]:
        """
        Load and chunk a text file
        
        Args:
            file_path (str): Path to text file
            chunk_size (int): Maximum size of each chunk
            
        Returns:
            List[str]: List of text chunks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split text into paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # Combine paragraphs into chunks of approximately chunk_size
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) <= chunk_size:
                    current_chunk += paragraph + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        except Exception as e:
            print(f"Error loading text file {file_path}: {e}")
            return []
    
    def fetch_web_content(self, url: str, max_retries: int = 3) -> Optional[str]:
        """
        Fetch content from a web page
        
        Args:
            url (str): URL to fetch
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            Optional[str]: HTML content or None if failed
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                return response.text
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {url} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def extract_text_from_html(self, html: str) -> str:
        """
        Extract main text content from HTML
        
        Args:
            html (str): HTML content
            
        Returns:
            str: Extracted text
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
            
            # Get text
            text = soup.get_text(separator=' ')
            
            # Clean up text
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = ' '.join(lines)
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            print(f"Error extracting text from HTML: {e}")
            return ""
    
    def process_web_page(self, url: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Process a web page and extract content
        
        Args:
            url (str): URL to process
            chunk_size (int): Maximum size of each chunk
            
        Returns:
            List[Dict[str, Any]]: List of document chunks with metadata
        """
        html = self.fetch_web_content(url)
        if not html:
            return []
        
        text = self.extract_text_from_html(html)
        if not text:
            return []
        
        # Split text into chunks
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) <= chunk_size:
                current_chunk.append(word)
                current_length += len(word) + 1  # +1 for space
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Create document objects
        domain = urlparse(url).netloc
        documents = []
        
        for i, chunk in enumerate(chunks):
            documents.append({
                'content': chunk,
                'metadata': {
                    'source': 'web',
                    'url': url,
                    'domain': domain,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            })
        
        return documents
    
    def save_documents(self, documents: List[Dict[str, Any]], file_name: str) -> str:
        """
        Save documents to a JSON file
        
        Args:
            documents (List[Dict[str, Any]]): Documents to save
            file_name (str): Output file name
            
        Returns:
            str: Path to saved file
        """
        file_path = os.path.join(self.base_dir, file_name)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2)
            print(f"Saved {len(documents)} documents to {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving documents to {file_path}: {e}")
            return ""
    
    def load_documents_from_directory(self, directory: str) -> Dict[str, List[Union[str, Dict[str, Any]]]]:
        """
        Load all documents from a directory
        
        Args:
            directory (str): Directory path
            
        Returns:
            Dict[str, List[Union[str, Dict[str, Any]]]]: Dictionary of document lists by type
        """
        documents = {
            'text': [],
            'json': [],
            'web': []
        }
        
        try:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                
                if not os.path.isfile(file_path):
                    continue
                
                if filename.endswith('.txt'):
                    text_chunks = self.load_text_file(file_path)
                    documents['text'].extend(text_chunks)
                
                elif filename.endswith('.json'):
                    json_docs = self.load_json_file(file_path)
                    documents['json'].extend(json_docs)
        
        except Exception as e:
            print(f"Error loading documents from directory {directory}: {e}")
        
        return documents
    
    def process_wikipedia_topic(self, topic: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Fetch and process Wikipedia articles on a topic
        
        Args:
            topic (str): Topic to search for
            max_results (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of document chunks with metadata
        """
        # Search Wikipedia
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={topic}&format=json&utf8=1"
        
        try:
            response = requests.get(search_url)
            response.raise_for_status()
            data = response.json()
            
            search_results = data.get('query', {}).get('search', [])
            documents = []
            
            for i, result in enumerate(search_results[:max_results]):
                page_id = result.get('pageid')
                title = result.get('title')
                
                if not page_id or not title:
                    continue
                
                # Get full article content
                content_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&pageids={page_id}&format=json&utf8=1"
                content_response = requests.get(content_url)
                content_response.raise_for_status()
                content_data = content_response.json()
                
                extract = content_data.get('query', {}).get('pages', {}).get(str(page_id), {}).get('extract', '')
                
                if extract:
                    documents.append({
                        'content': extract,
                        'metadata': {
                            'source': 'wikipedia',
                            'title': title,
                            'page_id': page_id,
                            'topic': topic
                        }
                    })
            
            return documents
        
        except Exception as e:
            print(f"Error processing Wikipedia topic {topic}: {e}")
            return []


# Example usage
if __name__ == "__main__":
    loader = DocumentLoader()
    
    # Test loading a web page
    web_docs = loader.process_web_page("https://en.wikipedia.org/wiki/Artificial_intelligence")
    print(f"Loaded {len(web_docs)} chunks from web page")
    
    # Test Wikipedia topic
    wiki_docs = loader.process_wikipedia_topic("machine learning")
    print(f"Loaded {len(wiki_docs)} Wikipedia articles")
    
    # Save documents
    if web_docs:
        loader.save_documents(web_docs, "web_ai_content.json")
    
    if wiki_docs:
        loader.save_documents(wiki_docs, "wiki_ml_content.json")
