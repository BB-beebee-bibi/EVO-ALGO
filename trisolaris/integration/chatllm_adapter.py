"""
ChatLLM adapter for interacting with the Abacus.AI API.
"""
import requests
from typing import List, Dict
import json

class ChatLLMAdapter:
    """Adapter for interacting with the ChatLLM API."""
    
    def __init__(self):
        """Initialize the ChatLLM adapter with the provided API key."""
        self.api_key = "s2_9096343ff8864daea0a036400d81eb4e"
        self.api_base_url = "https://api.abacus.ai/chat/completions"  # Simplified endpoint
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         model: str = "claude-3-sonnet-20240229",
                         max_tokens: int = 1000,
                         temperature: float = 0.7) -> str:
        """
        Generate a response from the ChatLLM API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Print request details for debugging
        print(f"Making request to: {self.api_base_url}")
        print(f"Headers: {json.dumps(self.headers, indent=2)}")
        print(f"Data: {json.dumps(data, indent=2)}")
        
        response = requests.post(
            self.api_base_url,
            headers=self.headers,
            json=data,
            timeout=30  # Add timeout
        )
        
        # Print response details for debugging
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response body: {response.text[:500]}...")  # Print first 500 chars
        
        response_data = response.json()
        return response_data['choices'][0]['message']['content']

    def validate_connection(self) -> bool:
        """
        Validate the connection to the ChatLLM API.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            response = self.generate_response(
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=10
            )
            return response is not None
        except Exception as e:
            print(f"Connection validation error: {e}")
            return False 