"""
Model integrations for the RAG Transformer system.
This module provides interfaces to various language models.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BaseLanguageModel(ABC):
    """Abstract base class for language model integrations"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
        """
        Generate text based on the prompt
        
        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            Dict[str, Any]: Generated text and metadata
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model
        
        Returns:
            Dict[str, Any]: Model information
        """
        pass


class HuggingFaceModel(BaseLanguageModel):
    """Integration with Hugging Face models"""
    
    def __init__(self, model_name: str = 'google/flan-t5-small'):
        """
        Initialize Hugging Face model
        
        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print(f"Loaded Hugging Face model: {model_name}")
        except Exception as e:
            print(f"Error loading Hugging Face model {model_name}: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
        """
        Generate text using Hugging Face model
        
        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            Dict[str, Any]: Generated text and metadata
        """
        start_time = time.time()
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    no_repeat_ngram_size=3
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            
            return {
                "text": response,
                "model": self.model_name,
                "generation_time": generation_time,
                "token_count": len(outputs[0]),
                "provider": "Hugging Face"
            }
            
        except Exception as e:
            print(f"Error generating text with Hugging Face model: {e}")
            return {
                "text": f"Error generating text: {str(e)}",
                "model": self.model_name,
                "generation_time": time.time() - start_time,
                "error": str(e),
                "provider": "Hugging Face"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Hugging Face model
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "name": self.model_name,
            "provider": "Hugging Face",
            "type": "local",
            "parameters": "unknown",  # Could be determined based on model name
            "capabilities": ["text generation"]
        }


class OpenAIModel(BaseLanguageModel):
    """Integration with OpenAI models"""
    
    def __init__(self, model_name: str = 'gpt-3.5-turbo'):
        """
        Initialize OpenAI model
        
        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            print(f"Initialized OpenAI client for model: {model_name}")
        except ImportError:
            print("Error: openai package not installed. Install with 'pip install openai'")
            raise
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
        """
        Generate text using OpenAI model
        
        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            Dict[str, Any]: Generated text and metadata
        """
        if not self.api_key:
            return {
                "text": "Error: OPENAI_API_KEY not set in environment variables",
                "model": self.model_name,
                "generation_time": 0,
                "error": "API key not set",
                "provider": "OpenAI"
            }
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate and concise information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            generation_time = time.time() - start_time
            
            return {
                "text": response.choices[0].message.content,
                "model": self.model_name,
                "generation_time": generation_time,
                "token_count": response.usage.total_tokens,
                "provider": "OpenAI",
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
            
        except Exception as e:
            print(f"Error generating text with OpenAI model: {e}")
            return {
                "text": f"Error generating text: {str(e)}",
                "model": self.model_name,
                "generation_time": time.time() - start_time,
                "error": str(e),
                "provider": "OpenAI"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the OpenAI model
        
        Returns:
            Dict[str, Any]: Model information
        """
        model_parameters = {
            "gpt-3.5-turbo": "175B",
            "gpt-4": "1.5T",
            "gpt-4-turbo": "1.8T"
        }
        
        return {
            "name": self.model_name,
            "provider": "OpenAI",
            "type": "api",
            "parameters": model_parameters.get(self.model_name, "unknown"),
            "capabilities": ["text generation", "reasoning", "instruction following"]
        }


class AnthropicModel(BaseLanguageModel):
    """Integration with Anthropic Claude models"""
    
    def __init__(self, model_name: str = 'claude-3-sonnet-20240229'):
        """
        Initialize Anthropic model
        
        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if not self.api_key:
            print("Warning: ANTHROPIC_API_KEY not found in environment variables")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            print(f"Initialized Anthropic client for model: {model_name}")
        except ImportError:
            print("Error: anthropic package not installed. Install with 'pip install anthropic'")
            raise
        except Exception as e:
            print(f"Error initializing Anthropic client: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 150) -> Dict[str, Any]:
        """
        Generate text using Anthropic model
        
        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            Dict[str, Any]: Generated text and metadata
        """
        if not self.api_key:
            return {
                "text": "Error: ANTHROPIC_API_KEY not set in environment variables",
                "model": self.model_name,
                "generation_time": 0,
                "error": "API key not set",
                "provider": "Anthropic"
            }
        
        start_time = time.time()
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                system="You are a helpful assistant that provides accurate and concise information.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            generation_time = time.time() - start_time
            
            return {
                "text": response.content[0].text,
                "model": self.model_name,
                "generation_time": generation_time,
                "provider": "Anthropic",
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
            
        except Exception as e:
            print(f"Error generating text with Anthropic model: {e}")
            return {
                "text": f"Error generating text: {str(e)}",
                "model": self.model_name,
                "generation_time": time.time() - start_time,
                "error": str(e),
                "provider": "Anthropic"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Anthropic model
        
        Returns:
            Dict[str, Any]: Model information
        """
        model_parameters = {
            "claude-3-opus-20240229": "~1T",
            "claude-3-sonnet-20240229": "~500B",
            "claude-3-haiku-20240307": "~100B"
        }
        
        return {
            "name": self.model_name,
            "provider": "Anthropic",
            "type": "api",
            "parameters": model_parameters.get(self.model_name, "unknown"),
            "capabilities": ["text generation", "reasoning", "instruction following"]
        }


class ModelFactory:
    """Factory for creating language model instances"""
    
    @staticmethod
    def create_model(model_type: str, model_name: Optional[str] = None) -> BaseLanguageModel:
        """
        Create a language model instance
        
        Args:
            model_type (str): Type of model ('huggingface', 'openai', 'anthropic')
            model_name (Optional[str]): Name of the model to use
            
        Returns:
            BaseLanguageModel: Language model instance
        """
        if model_type.lower() == 'huggingface':
            return HuggingFaceModel(model_name or 'google/flan-t5-small')
        elif model_type.lower() == 'openai':
            return OpenAIModel(model_name or 'gpt-3.5-turbo')
        elif model_type.lower() == 'anthropic':
            return AnthropicModel(model_name or 'claude-3-sonnet-20240229')
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Example usage
if __name__ == "__main__":
    # Test Hugging Face model
    hf_model = ModelFactory.create_model('huggingface', 'google/flan-t5-small')
    hf_response = hf_model.generate("What is machine learning?")
    print(f"Hugging Face response: {hf_response['text']}")
    print(f"Generation time: {hf_response['generation_time']:.2f} seconds")
    
    # Test OpenAI model if API key is available
    if os.getenv('OPENAI_API_KEY'):
        openai_model = ModelFactory.create_model('openai')
        openai_response = openai_model.generate("What is machine learning?")
        print(f"\nOpenAI response: {openai_response['text']}")
        print(f"Generation time: {openai_response['generation_time']:.2f} seconds")
    
    # Test Anthropic model if API key is available
    if os.getenv('ANTHROPIC_API_KEY'):
        anthropic_model = ModelFactory.create_model('anthropic')
        anthropic_response = anthropic_model.generate("What is machine learning?")
        print(f"\nAnthropic response: {anthropic_response['text']}")
        print(f"Generation time: {anthropic_response['generation_time']:.2f} seconds")
