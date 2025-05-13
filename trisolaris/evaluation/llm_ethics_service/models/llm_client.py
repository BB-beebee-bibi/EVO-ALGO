"""
Abstract base class for LLM clients.

This module defines the interface for all LLM clients used by the ethics service.
Each specific LLM implementation (Claude, GPT, etc.) should inherit from this
base class and implement its abstract methods.
"""

import abc
import logging
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMClientError(Exception):
    """Exception raised when an LLM client encounters an error."""
    pass

class LLMClient(abc.ABC):
    """
    Abstract base class for LLM clients.
    
    This class defines the interface that all LLM client implementations
    must adhere to, ensuring consistent behavior across different models.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key for the LLM service
            model: Model identifier to use
        """
        self.api_key = api_key
        self.model = model
        logger.info(f"Initialized {self.__class__.__name__} client")
    
    @abc.abstractmethod
    async def evaluate_ethics(self, code: str) -> Dict[str, Any]:
        """
        Evaluate the ethics of code using the LLM.
        
        Args:
            code: The source code to evaluate
            
        Returns:
            Dictionary containing the ethics evaluation report
            
        Raises:
            LLMClientError: If the evaluation fails
        """
        pass
    
    @abc.abstractmethod
    async def evaluate_gurbani_alignment(self, code: str, ethics_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate how well code aligns with Gurbani principles.
        
        Args:
            code: The source code to evaluate
            ethics_report: The general ethics report for context
            
        Returns:
            Dictionary containing the Gurbani alignment evaluation
            
        Raises:
            LLMClientError: If the evaluation fails
        """
        pass
    
    @abc.abstractmethod
    async def get_improvement_suggestions(self, code: str, ethics_report: Dict[str, Any]) -> List[str]:
        """
        Get suggestions for improving code ethics.
        
        Args:
            code: The source code to evaluate
            ethics_report: The ethics report containing issues to address
            
        Returns:
            List of improvement suggestions
            
        Raises:
            LLMClientError: If generating suggestions fails
        """
        pass
    
    @abc.abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the LLM service is available and responding.
        
        Returns:
            True if the service is healthy, False otherwise
        """
        pass
    
    def _validate_response(self, response: Dict[str, Any], required_fields: List[str]) -> None:
        """
        Validate that an LLM response contains all required fields.
        
        Args:
            response: The response to validate
            required_fields: List of field names that must be present
            
        Raises:
            LLMClientError: If any required field is missing
        """
        missing_fields = [field for field in required_fields if field not in response]
        if missing_fields:
            error_msg = f"LLM response missing required fields: {', '.join(missing_fields)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
    
    def _format_ethics_prompt(self, code: str) -> str:
        """
        Format the prompt for ethics evaluation.
        
        Args:
            code: The source code to evaluate
            
        Returns:
            Formatted prompt string
        """
        return f"""
        You are an expert in ethical code evaluation. Please analyze the following code for ethical concerns
        related to privacy, security, fairness, transparency, and accountability.
        
        Provide a detailed ethics evaluation report with the following structure:
        1. Overall ethics score (0.0 to 1.0)
        2. Category scores for: privacy, security, fairness, transparency, accountability
        3. Potential ethical issues identified
        4. Recommendations for addressing any concerns
        
        Code to evaluate:
        ```python
        {code}
        ```
        
        Format your response as a JSON object with the following structure:
        {{
            "overall_score": float,
            "categories": {{
                "privacy": {{"score": float, "concerns": [string], "recommendations": [string]}},
                "security": {{"score": float, "concerns": [string], "recommendations": [string]}},
                "fairness": {{"score": float, "concerns": [string], "recommendations": [string]}},
                "transparency": {{"score": float, "concerns": [string], "recommendations": [string]}},
                "accountability": {{"score": float, "concerns": [string], "recommendations": [string]}}
            }},
            "potential_issues": [
                {{"category": string, "severity": string, "description": string}}
            ],
            "recommendations": [string]
        }}
        """
    
    def _format_gurbani_prompt(self, code: str, ethics_report: Dict[str, Any], principles: str) -> str:
        """
        Format the prompt for Gurbani alignment evaluation.
        
        Args:
            code: The source code to evaluate
            ethics_report: The general ethics report for context
            principles: The Gurbani principles text
            
        Returns:
            Formatted prompt string
        """
        return f"""
        You are an expert in evaluating code against Gurbani principles. Please analyze the following code
        and determine how well it aligns with Gurbani principles.
        
        Here are the Gurbani principles to consider:
        {principles}
        
        The code has already undergone a general ethics evaluation with the following results:
        Overall ethics score: {ethics_report.get('overall_score', 'N/A')}
        
        Code to evaluate:
        ```python
        {code}
        ```
        
        Please evaluate how well this code aligns with each of the Gurbani principles.
        Format your response as a JSON object with the following structure:
        {{
            "score": float,  // Overall alignment score from 0.0 to 1.0
            "principle_scores": {{
                "unity_in_design": float,
                "natural_flow": float,
                "truth_and_transparency": float,
                "service_oriented": float,
                "balance_and_harmony": float,
                "ego_free_development": float,
                "universal_design": float,
                "mindful_resource_usage": float
            }},
            "concerns": [string],  // List of specific concerns
            "recommendations": [string]  // List of recommendations for better alignment
        }}
        """
    
    def _format_suggestions_prompt(self, code: str, ethics_report: Dict[str, Any]) -> str:
        """
        Format the prompt for improvement suggestions.
        
        Args:
            code: The source code to evaluate
            ethics_report: The ethics report containing issues to address
            
        Returns:
            Formatted prompt string
        """
        # Extract issues from the ethics report
        issues = []
        if 'potential_issues' in ethics_report:
            issues = [f"{issue.get('category', 'unknown')}: {issue.get('description', 'No description')}"
                     for issue in ethics_report.get('potential_issues', [])]
        
        issues_text = "\n".join([f"- {issue}" for issue in issues]) if issues else "No specific issues identified."
        
        return f"""
        You are an expert in ethical code improvement. The following code has undergone an ethics evaluation
        and the following issues were identified:
        
        {issues_text}
        
        Code to improve:
        ```python
        {code}
        ```
        
        Please provide specific, actionable suggestions for improving this code to address the ethical concerns.
        Format your response as a JSON array of strings, each containing one improvement suggestion.
        """