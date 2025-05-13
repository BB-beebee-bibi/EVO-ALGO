"""
Claude LLM client implementation.

This module provides a concrete implementation of the LLM client interface
for Anthropic's Claude models, handling API communication, response parsing,
and error handling specific to Claude.
"""

import json
import logging
import os
import httpx
import asyncio
from typing import Dict, Any, Optional, List, Union

from ..config import settings
from .llm_client import LLMClient, LLMClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClaudeClient(LLMClient):
    """
    Claude LLM client implementation.
    
    This class implements the LLMClient interface for Anthropic's Claude models,
    providing methods to evaluate code ethics using Claude's capabilities.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Claude client.
        
        Args:
            api_key: Claude API key (defaults to config if not provided)
            model: Claude model to use (defaults to config if not provided)
        """
        # Use provided values or defaults from settings
        api_key = api_key or settings.CLAUDE_API_KEY
        model = model or settings.CLAUDE_MODEL
        
        if not api_key:
            logger.warning("No Claude API key provided. Set CLAUDE_API_KEY in environment or config.")
        
        super().__init__(api_key=api_key, model=model)
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.client = httpx.AsyncClient(timeout=60.0)  # 60 second timeout
        
        # Add API key to headers
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        logger.info(f"Initialized Claude client with model: {self.model}")
    
    async def evaluate_ethics(self, code: str) -> Dict[str, Any]:
        """
        Evaluate the ethics of code using Claude.
        
        Args:
            code: The source code to evaluate
            
        Returns:
            Dictionary containing the ethics evaluation report
            
        Raises:
            LLMClientError: If the evaluation fails
        """
        try:
            prompt = self._format_ethics_prompt(code)
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "max_tokens": 4000,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1  # Low temperature for more deterministic responses
            }
            
            # Make the API request
            logger.info("Sending ethics evaluation request to Claude")
            response = await self.client.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Extract the content from Claude's response
            if "content" not in response_data or not response_data["content"]:
                raise LLMClientError("Empty response from Claude API")
            
            # Extract the JSON from the response
            content = response_data["content"][0]["text"]
            
            # Find JSON in the response (Claude might wrap it in markdown code blocks)
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                raise LLMClientError("Could not find JSON in Claude response")
            
            json_str = content[json_start:json_end]
            
            # Parse the JSON
            ethics_report = json.loads(json_str)
            
            # Validate the response structure
            required_fields = ["overall_score", "categories", "potential_issues", "recommendations"]
            self._validate_response(ethics_report, required_fields)
            
            logger.info(f"Ethics evaluation completed with score: {ethics_report['overall_score']}")
            return ethics_report
            
        except httpx.HTTPError as e:
            error_msg = f"HTTP error during Claude ethics evaluation: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Claude response as JSON: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
        except Exception as e:
            error_msg = f"Error during Claude ethics evaluation: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
    
    async def evaluate_gurbani_alignment(self, code: str, ethics_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate how well code aligns with Gurbani principles using Claude.
        
        Args:
            code: The source code to evaluate
            ethics_report: The general ethics report for context
            
        Returns:
            Dictionary containing the Gurbani alignment evaluation
            
        Raises:
            LLMClientError: If the evaluation fails
        """
        try:
            # Load Gurbani principles
            principles_path = os.path.join(os.path.dirname(__file__), settings.GURBANI_PRINCIPLES_PATH)
            try:
                with open(principles_path, 'r') as f:
                    principles = f.read()
            except FileNotFoundError:
                logger.warning(f"Gurbani principles file not found at {principles_path}. Using default principles.")
                principles = """
                # Gurbani-Inspired Design Principles
                
                1. Unity in Design: Create systems that recognize and respect interconnection.
                2. Natural Flow: Work with rather than against natural patterns and limitations.
                3. Truth and Transparency: Create systems that embody and facilitate truthfulness.
                4. Service-Oriented Architecture: Design with the sincere intention to serve users.
                5. Balance and Harmony: Find the middle path between competing concerns.
                6. Ego-Free Development: Create with awareness of how technology can transcend ego.
                7. Universal Design: Create technology accessible and beneficial to all.
                8. Mindful Resource Usage: Use only what is necessary, without excess or waste.
                """
            
            prompt = self._format_gurbani_prompt(code, ethics_report, principles)
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "max_tokens": 4000,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1  # Low temperature for more deterministic responses
            }
            
            # Make the API request
            logger.info("Sending Gurbani alignment evaluation request to Claude")
            response = await self.client.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Extract the content from Claude's response
            if "content" not in response_data or not response_data["content"]:
                raise LLMClientError("Empty response from Claude API")
            
            # Extract the JSON from the response
            content = response_data["content"][0]["text"]
            
            # Find JSON in the response (Claude might wrap it in markdown code blocks)
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                raise LLMClientError("Could not find JSON in Claude response")
            
            json_str = content[json_start:json_end]
            
            # Parse the JSON
            gurbani_report = json.loads(json_str)
            
            # Validate the response structure
            required_fields = ["score", "principle_scores", "concerns", "recommendations"]
            self._validate_response(gurbani_report, required_fields)
            
            logger.info(f"Gurbani alignment evaluation completed with score: {gurbani_report['score']}")
            return gurbani_report
            
        except httpx.HTTPError as e:
            error_msg = f"HTTP error during Claude Gurbani evaluation: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Claude response as JSON: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
        except Exception as e:
            error_msg = f"Error during Claude Gurbani evaluation: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
    
    async def get_improvement_suggestions(self, code: str, ethics_report: Dict[str, Any]) -> List[str]:
        """
        Get suggestions for improving code ethics using Claude.
        
        Args:
            code: The source code to evaluate
            ethics_report: The ethics report containing issues to address
            
        Returns:
            List of improvement suggestions
            
        Raises:
            LLMClientError: If generating suggestions fails
        """
        try:
            prompt = self._format_suggestions_prompt(code, ethics_report)
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "max_tokens": 2000,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2  # Slightly higher temperature for creative suggestions
            }
            
            # Make the API request
            logger.info("Sending improvement suggestions request to Claude")
            response = await self.client.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            
            # Extract the content from Claude's response
            if "content" not in response_data or not response_data["content"]:
                raise LLMClientError("Empty response from Claude API")
            
            # Extract the JSON from the response
            content = response_data["content"][0]["text"]
            
            # Find JSON in the response (Claude might wrap it in markdown code blocks)
            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            
            if json_start == -1 or json_end == 0:
                # If we can't find a JSON array, try to extract suggestions from text
                logger.warning("Could not find JSON array in Claude response, extracting suggestions from text")
                lines = content.split("\n")
                suggestions = [line.strip() for line in lines if line.strip().startswith("- ")]
                if not suggestions:
                    suggestions = [content.strip()]
                return suggestions
            
            json_str = content[json_start:json_end]
            
            # Parse the JSON
            suggestions = json.loads(json_str)
            
            if not isinstance(suggestions, list):
                raise LLMClientError("Claude response is not a list of suggestions")
            
            logger.info(f"Generated {len(suggestions)} improvement suggestions")
            return suggestions
            
        except httpx.HTTPError as e:
            error_msg = f"HTTP error during Claude suggestions generation: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Claude response as JSON: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
        except Exception as e:
            error_msg = f"Error during Claude suggestions generation: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
    
    async def health_check(self) -> bool:
        """
        Check if the Claude API is available and responding.
        
        Returns:
            True if the service is healthy, False otherwise
        """
        try:
            # Simple health check with minimal tokens
            payload = {
                "model": self.model,
                "max_tokens": 10,
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }
            
            response = await self.client.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Claude health check failed: {str(e)}")
            return False