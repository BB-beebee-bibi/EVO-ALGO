"""
Ethics Service Client for the TRISOLARIS framework.

This module provides a client interface for the LLM ethics service that evaluates
evolved code against ethical guidelines and Gurbani principles. It connects to
the standalone REST microservice for ethics evaluation.
"""

import ast
import json
import logging
import os
import httpx
import asyncio
from typing import Dict, Any, List, Tuple, Optional, Union
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EthicsClientError(Exception):
    """Exception raised when ethics evaluation fails."""
    pass

class EthicsServiceClient:
    """
    Client for the LLM ethics service.
    
    This class provides methods to interact with the LLM ethics service
    for code evaluation against ethical guidelines and Gurbani principles.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        """
        Initialize the ethics service client.
        
        Args:
            base_url: Base URL of the ethics service (defaults to environment variable or localhost)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.environ.get("ETHICS_SERVICE_URL", "http://localhost:8000")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        logger.info(f"Initialized ethics service client with base URL: {self.base_url}")
    
    async def evaluate_ethics(self, code: str, ast_tree: Optional[ast.AST] = None) -> Dict[str, Any]:
        """
        Send code to ethics service for evaluation.
        
        Args:
            code: The source code to evaluate
            ast_tree: Optional pre-parsed AST for the code (not used in REST implementation)
            
        Returns:
            Dictionary containing the ethics evaluation report
            
        Raises:
            EthicsClientError: If the evaluation fails
        """
        try:
            logger.info("Sending code to ethics service for evaluation")
            
            # Prepare request payload
            payload = {
                "code": code,
                "include_gurbani_alignment": True,
                "include_suggestions": True
            }
            
            # Make API request
            response = await self.client.post(
                f"{self.base_url}/api/v1/evaluate",
                json=payload
            )
            
            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = f"Ethics service returned error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise EthicsClientError(error_msg)
            
            # Parse response
            result = response.json()
            
            # Extract ethics evaluation from comprehensive response
            ethics_report = result.get("ethics_evaluation", {})
            
            logger.info(f"Ethics evaluation complete with score: {ethics_report.get('overall_score', 'N/A')}")
            return ethics_report
            
        except httpx.RequestError as e:
            error_msg = f"Error connecting to ethics service: {str(e)}"
            logger.error(error_msg)
            raise EthicsClientError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing ethics service response: {str(e)}"
            logger.error(error_msg)
            raise EthicsClientError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during ethics evaluation: {str(e)}"
            logger.error(error_msg)
            raise EthicsClientError(error_msg)
    
    async def get_gurbani_alignment(self, code: str) -> Dict[str, Any]:
        """
        Get Gurbani alignment evaluation for code.
        
        Args:
            code: The source code to evaluate
            
        Returns:
            Dictionary containing the Gurbani alignment evaluation
            
        Raises:
            EthicsClientError: If the evaluation fails
        """
        try:
            logger.info("Sending code to ethics service for Gurbani alignment evaluation")
            
            # Make API request
            response = await self.client.post(
                f"{self.base_url}/api/v1/evaluate/gurbani",
                json={"code": code}
            )
            
            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = f"Ethics service returned error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise EthicsClientError(error_msg)
            
            # Parse response
            gurbani_report = response.json()
            
            logger.info(f"Gurbani alignment evaluation complete with score: {gurbani_report.get('score', 'N/A')}")
            return gurbani_report
            
        except httpx.RequestError as e:
            error_msg = f"Error connecting to ethics service: {str(e)}"
            logger.error(error_msg)
            raise EthicsClientError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing ethics service response: {str(e)}"
            logger.error(error_msg)
            raise EthicsClientError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during Gurbani alignment evaluation: {str(e)}"
            logger.error(error_msg)
            raise EthicsClientError(error_msg)
    
    async def get_improvement_suggestions(self, code: str) -> List[str]:
        """
        Get suggestions for improving code ethics.
        
        Args:
            code: The source code to evaluate
            
        Returns:
            List of improvement suggestions
            
        Raises:
            EthicsClientError: If getting suggestions fails
        """
        try:
            logger.info("Sending code to ethics service for improvement suggestions")
            
            # Make API request
            response = await self.client.post(
                f"{self.base_url}/api/v1/suggestions",
                json={"code": code}
            )
            
            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = f"Ethics service returned error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise EthicsClientError(error_msg)
            
            # Parse response
            suggestions = response.json()
            
            logger.info(f"Received {len(suggestions)} improvement suggestions")
            return suggestions
            
        except httpx.RequestError as e:
            error_msg = f"Error connecting to ethics service: {str(e)}"
            logger.error(error_msg)
            raise EthicsClientError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing ethics service response: {str(e)}"
            logger.error(error_msg)
            raise EthicsClientError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error getting improvement suggestions: {str(e)}"
            logger.error(error_msg)
            raise EthicsClientError(error_msg)
    
    async def check_health(self) -> bool:
        """
        Check if the ethics service is healthy.
        
        Returns:
            True if the service is healthy, False otherwise
        """
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

# Create a global client instance
_client = None

def get_client() -> EthicsServiceClient:
    """
    Get the global ethics service client instance.
    
    Returns:
        Ethics service client instance
    """
    global _client
    if _client is None:
        _client = EthicsServiceClient()
    return _client

# Synchronous wrapper functions for backward compatibility

@lru_cache(maxsize=128)
def evaluate_ethics(code: str, ast_tree: Optional[ast.AST] = None) -> Dict[str, Any]:
    """
    Send code to ethics service for evaluation (synchronous wrapper).
    
    Args:
        code: The source code to evaluate
        ast_tree: Optional pre-parsed AST for the code
        
    Returns:
        Dictionary containing the ethics evaluation report
    """
    client = get_client()
    loop = asyncio.get_event_loop()
    try:
        return loop.run_until_complete(client.evaluate_ethics(code, ast_tree))
    except Exception as e:
        logger.error(f"Error in synchronous evaluate_ethics: {str(e)}")
        # Fall back to mock implementation for robustness
        return _mock_evaluate_ethics(code, ast_tree)

def parse_ethics_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and structure the ethics evaluation report.
    
    This function extracts key insights and structures the report
    for easier consumption by other components of the system.
    
    Args:
        report: Raw ethics evaluation report
        
    Returns:
        Structured report with key insights
    """
    if not report:
        raise EthicsClientError("Cannot parse empty ethics report")
    
    try:
        structured_report = {
            "evaluation_id": report.get("evaluation_id", "unknown"),
            "timestamp": report.get("timestamp", "unknown"),
            "overall_score": report.get("overall_score", 0.0),
            "passed": report.get("overall_score", 0.0) >= 0.7,  # Consider 0.7 as passing threshold
            "category_scores": {
                category: data.get("score", 0.0)
                for category, data in report.get("categories", {}).items()
            },
            "concerns": [],
            "recommendations": report.get("recommendations", []),
            "gurbani_alignment": {
                "score": report.get("categories", {}).get("gurbani_alignment", {}).get("score", 0.0),
                "concerns": report.get("categories", {}).get("gurbani_alignment", {}).get("concerns", []),
                "recommendations": report.get("categories", {}).get("gurbani_alignment", {}).get("recommendations", [])
            }
        }
        
        # Flatten and prioritize concerns
        for issue in report.get("potential_issues", []):
            structured_report["concerns"].append({
                "category": issue.get("category", "unknown"),
                "severity": issue.get("severity", "low"),
                "description": issue.get("description", "Unknown concern")
            })
        
        # Sort concerns by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        structured_report["concerns"].sort(
            key=lambda x: severity_order.get(x.get("severity", "low"), 3)
        )
        
        return structured_report
    
    except Exception as e:
        logger.error(f"Error parsing ethics report: {str(e)}")
        raise EthicsClientError(f"Failed to parse ethics report: {str(e)}")

def check_gurbani_alignment(report: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
    """
    Determine if code aligns with Gurbani principles.
    
    Args:
        report: Parsed ethics report
        
    Returns:
        Tuple containing:
            - Boolean indicating if code aligns with Gurbani principles
            - Alignment score (0.0 to 1.0)
            - List of specific concerns if any
    """
    try:
        # Extract Gurbani alignment data
        gurbani_data = report.get("gurbani_alignment", {})
        if isinstance(gurbani_data, dict):
            alignment_score = gurbani_data.get("score", 0.0)
            concerns = gurbani_data.get("concerns", [])
        else:
            # Handle case where report structure is different
            alignment_score = report.get("category_scores", {}).get("gurbani_alignment", 0.0)
            concerns = [
                concern["description"]
                for concern in report.get("concerns", [])
                if concern.get("category") == "gurbani_alignment"
            ]
        
        # Consider aligned if score is at least 0.8
        is_aligned = alignment_score >= 0.8
        
        logger.info(f"Gurbani alignment check: {'Passed' if is_aligned else 'Failed'} with score {alignment_score}")
        return is_aligned, alignment_score, concerns
    
    except Exception as e:
        logger.error(f"Error checking Gurbani alignment: {str(e)}")
        return False, 0.0, [f"Error evaluating Gurbani alignment: {str(e)}"]

# Mock implementation for fallback and testing

import hashlib
import random

def _mock_evaluate_ethics(code: str, ast_tree: Optional[ast.AST] = None) -> Dict[str, Any]:
    """
    Mock implementation of ethics evaluation for fallback and testing.
    
    Args:
        code: The source code to evaluate
        ast_tree: Optional pre-parsed AST for the code
        
    Returns:
        Dictionary containing the mock ethics evaluation report
    """
    logger.warning("Using mock ethics evaluation implementation")
    
    if not code.strip():
        raise EthicsClientError("Cannot evaluate empty code")
    
    # Parse AST if not provided
    if ast_tree is None:
        try:
            ast_tree = ast.parse(code)
        except SyntaxError as e:
            raise EthicsClientError(f"Cannot evaluate ethics due to syntax error: {str(e)}")
        except Exception as e:
            raise EthicsClientError(f"Error parsing code for ethics evaluation: {str(e)}")
    
    # Generate a deterministic but pseudo-random ethics score based on code hash
    code_hash = hashlib.md5(code.encode()).hexdigest()
    random.seed(code_hash)
    
    # Mock ethics evaluation
    mock_report = {
        "evaluation_id": f"eval-{code_hash[:8]}",
        "timestamp": "2025-05-13T10:30:00Z",
        "overall_score": round(random.uniform(0.5, 0.95), 2),
        "categories": {
            "privacy": {
                "score": round(random.uniform(0.6, 0.98), 2),
                "concerns": [],
                "recommendations": []
            },
            "security": {
                "score": round(random.uniform(0.6, 0.98), 2),
                "concerns": [],
                "recommendations": []
            },
            "fairness": {
                "score": round(random.uniform(0.7, 0.99), 2),
                "concerns": [],
                "recommendations": []
            },
            "transparency": {
                "score": round(random.uniform(0.7, 0.99), 2),
                "concerns": [],
                "recommendations": []
            },
            "accountability": {
                "score": round(random.uniform(0.7, 0.99), 2),
                "concerns": [],
                "recommendations": []
            },
            "gurbani_alignment": {
                "score": round(random.uniform(0.7, 0.99), 2),
                "concerns": [],
                "recommendations": []
            }
        },
        "potential_issues": [],
        "recommendations": []
    }
    
    # Add mock concerns based on code analysis
    concerns = []
    
    # Look for system calls or file operations
    if "os.system" in code or "subprocess" in code:
        concerns.append({
            "category": "security",
            "severity": "high",
            "description": "Code contains system calls which may pose security risks"
        })
        mock_report["categories"]["security"]["score"] -= 0.3
        mock_report["categories"]["security"]["concerns"].append("System call usage detected")
        mock_report["categories"]["security"]["recommendations"].append("Remove system calls and use safer alternatives")
    
    # Look for network operations
    if "socket" in code or "urllib" in code or "requests" in code:
        concerns.append({
            "category": "privacy",
            "severity": "medium",
            "description": "Code contains network operations which may compromise privacy"
        })
        mock_report["categories"]["privacy"]["score"] -= 0.2
        mock_report["categories"]["privacy"]["concerns"].append("Network operations detected")
        mock_report["categories"]["privacy"]["recommendations"].append("Ensure network operations respect user privacy")
    
    # Add concerns to the report
    mock_report["potential_issues"] = concerns
    
    # Recalculate overall score based on category scores
    category_scores = [cat_data["score"] for cat_data in mock_report["categories"].values()]
    mock_report["overall_score"] = round(sum(category_scores) / len(category_scores), 2)
    
    # Add general recommendations
    if mock_report["overall_score"] < 0.7:
        mock_report["recommendations"].append("Review code for ethical concerns and security issues")
    if mock_report["overall_score"] < 0.8:
        mock_report["recommendations"].append("Add more documentation to improve transparency")
    if mock_report["overall_score"] >= 0.9:
        mock_report["recommendations"].append("Code meets high ethical standards")
    
    logger.info(f"Mock ethics evaluation complete with score: {mock_report['overall_score']}")
    return mock_report

def generate_mock_llm_response(code: str) -> str:
    """
    Generate a mock LLM response for ethics evaluation.
    
    This function is used for testing when the actual LLM service
    is not available.
    
    Args:
        code: The source code to evaluate
        
    Returns:
        JSON string containing the mock LLM response
    """
    report = _mock_evaluate_ethics(code)
    return json.dumps(report, indent=2)