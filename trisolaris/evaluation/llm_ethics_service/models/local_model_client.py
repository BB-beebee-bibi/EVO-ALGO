"""
Local model client implementation.

This module provides a fallback implementation of the LLM client interface
that uses local heuristics and rule-based analysis when external LLM APIs
are not available or for testing purposes.
"""

import ast
import json
import logging
import re
import hashlib
import random
from typing import Dict, Any, Optional, List, Union

from ..config import settings
from .llm_client import LLMClient, LLMClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalModelClient(LLMClient):
    """
    Local model client implementation.
    
    This class implements the LLMClient interface using local heuristics
    and rule-based analysis rather than external API calls. It's useful
    for testing, offline operation, or as a fallback mechanism.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the local model client.
        
        Args:
            api_key: Not used for local model, included for interface compatibility
            model: Not used for local model, included for interface compatibility
        """
        super().__init__(api_key=None, model="local")
        logger.info("Initialized local model client")
    
    async def evaluate_ethics(self, code: str) -> Dict[str, Any]:
        """
        Evaluate the ethics of code using local heuristics.
        
        Args:
            code: The source code to evaluate
            
        Returns:
            Dictionary containing the ethics evaluation report
            
        Raises:
            LLMClientError: If the evaluation fails
        """
        try:
            if not code.strip():
                raise LLMClientError("Cannot evaluate empty code")
            
            # Parse AST for analysis
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                raise LLMClientError(f"Cannot evaluate ethics due to syntax error: {str(e)}")
            except Exception as e:
                raise LLMClientError(f"Error parsing code for ethics evaluation: {str(e)}")
            
            # Generate a deterministic but pseudo-random ethics score based on code hash
            # This ensures the same code always gets the same evaluation
            code_hash = hashlib.md5(code.encode()).hexdigest()
            random.seed(code_hash)
            
            # Initialize ethics report
            ethics_report = {
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
            
            # Check for potentially concerning patterns
            concerns = []
            
            # Look for system calls or file operations
            if "os.system" in code or "subprocess" in code:
                concerns.append({
                    "category": "security",
                    "severity": "high",
                    "description": "Code contains system calls which may pose security risks"
                })
                ethics_report["categories"]["security"]["score"] -= 0.3
                ethics_report["categories"]["security"]["concerns"].append("System call usage detected")
                ethics_report["categories"]["security"]["recommendations"].append("Remove system calls and use safer alternatives")
            
            # Look for network operations
            if "socket" in code or "urllib" in code or "requests" in code:
                concerns.append({
                    "category": "privacy",
                    "severity": "medium",
                    "description": "Code contains network operations which may compromise privacy"
                })
                ethics_report["categories"]["privacy"]["score"] -= 0.2
                ethics_report["categories"]["privacy"]["concerns"].append("Network operations detected")
                ethics_report["categories"]["privacy"]["recommendations"].append("Ensure network operations respect user privacy")
            
            # Look for file operations
            if "open(" in code or "write(" in code or "read(" in code:
                concerns.append({
                    "category": "security",
                    "severity": "medium",
                    "description": "Code contains file operations which may pose security risks"
                })
                ethics_report["categories"]["security"]["score"] -= 0.2
                ethics_report["categories"]["security"]["concerns"].append("File operations detected")
                ethics_report["categories"]["security"]["recommendations"].append("Ensure file operations are properly sandboxed")
            
            # Check for potentially harmful function names
            harmful_terms = ["hack", "exploit", "attack", "steal", "bypass", "crack"]
            for term in harmful_terms:
                if term in code.lower():
                    concerns.append({
                        "category": "ethics",
                        "severity": "high",
                        "description": f"Code contains potentially harmful term: {term}"
                    })
                    ethics_report["categories"]["gurbani_alignment"]["score"] -= 0.3
                    ethics_report["categories"]["gurbani_alignment"]["concerns"].append(f"Potentially harmful term detected: {term}")
                    ethics_report["categories"]["gurbani_alignment"]["recommendations"].append("Remove or rename functions with harmful terminology")
            
            # Add concerns to the report
            ethics_report["potential_issues"] = concerns
            
            # Recalculate overall score based on category scores
            category_scores = [cat_data["score"] for cat_data in ethics_report["categories"].values()]
            ethics_report["overall_score"] = round(sum(category_scores) / len(category_scores), 2)
            
            # Add general recommendations
            if ethics_report["overall_score"] < 0.7:
                ethics_report["recommendations"].append("Review code for ethical concerns and security issues")
            if ethics_report["overall_score"] < 0.8:
                ethics_report["recommendations"].append("Add more documentation to improve transparency")
            if ethics_report["overall_score"] >= 0.9:
                ethics_report["recommendations"].append("Code meets high ethical standards")
            
            logger.info(f"Local ethics evaluation completed with score: {ethics_report['overall_score']}")
            return ethics_report
            
        except Exception as e:
            error_msg = f"Error during local ethics evaluation: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
    
    async def evaluate_gurbani_alignment(self, code: str, ethics_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate how well code aligns with Gurbani principles using local heuristics.
        
        Args:
            code: The source code to evaluate
            ethics_report: The general ethics report for context
            
        Returns:
            Dictionary containing the Gurbani alignment evaluation
            
        Raises:
            LLMClientError: If the evaluation fails
        """
        try:
            # Generate a deterministic but pseudo-random alignment score based on code hash
            code_hash = hashlib.md5(code.encode()).hexdigest()
            random.seed(code_hash)
            
            # Base alignment score on ethics report
            base_score = ethics_report.get("overall_score", 0.7)
            
            # Initialize Gurbani alignment report
            gurbani_report = {
                "score": round(min(base_score + random.uniform(-0.1, 0.1), 1.0), 2),
                "principle_scores": {
                    "unity_in_design": round(random.uniform(0.6, 0.95), 2),
                    "natural_flow": round(random.uniform(0.6, 0.95), 2),
                    "truth_and_transparency": round(random.uniform(0.6, 0.95), 2),
                    "service_oriented": round(random.uniform(0.6, 0.95), 2),
                    "balance_and_harmony": round(random.uniform(0.6, 0.95), 2),
                    "ego_free_development": round(random.uniform(0.6, 0.95), 2),
                    "universal_design": round(random.uniform(0.6, 0.95), 2),
                    "mindful_resource_usage": round(random.uniform(0.6, 0.95), 2)
                },
                "concerns": [],
                "recommendations": []
            }
            
            # Check for specific Gurbani principle violations
            
            # Unity in Design: Check for excessive coupling or fragmentation
            if "import" in code and code.count("import") > 10:
                gurbani_report["principle_scores"]["unity_in_design"] -= 0.2
                gurbani_report["concerns"].append("Code has excessive imports, potentially violating Unity in Design")
                gurbani_report["recommendations"].append("Consolidate imports and reduce dependencies")
            
            # Natural Flow: Check for complex control flow
            if code.count("if ") > 15 or code.count("for ") > 10:
                gurbani_report["principle_scores"]["natural_flow"] -= 0.2
                gurbani_report["concerns"].append("Code has complex control flow, potentially violating Natural Flow")
                gurbani_report["recommendations"].append("Simplify control flow and break down complex functions")
            
            # Truth and Transparency: Check for documentation
            if code.count('"""') < 2 and code.count("def ") > 3:
                gurbani_report["principle_scores"]["truth_and_transparency"] -= 0.2
                gurbani_report["concerns"].append("Code lacks documentation, potentially violating Truth and Transparency")
                gurbani_report["recommendations"].append("Add docstrings and comments to improve transparency")
            
            # Service-Oriented: Check for user-focused error handling
            if "except" in code and "error_msg" not in code.lower():
                gurbani_report["principle_scores"]["service_oriented"] -= 0.1
                gurbani_report["concerns"].append("Code lacks user-friendly error messages, potentially violating Service-Oriented principle")
                gurbani_report["recommendations"].append("Add clear error messages to help users understand issues")
            
            # Balance and Harmony: Check for very long functions
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_lines = len(ast.unparse(node).split("\n"))
                        if func_lines > 50:
                            gurbani_report["principle_scores"]["balance_and_harmony"] -= 0.2
                            gurbani_report["concerns"].append("Code has very long functions, potentially violating Balance and Harmony")
                            gurbani_report["recommendations"].append("Break down long functions into smaller, focused components")
                            break
            except:
                # Ignore AST parsing errors for this check
                pass
            
            # Ego-Free Development: Check for hardcoded credentials or personal identifiers
            if "password" in code.lower() or "api_key" in code.lower():
                gurbani_report["principle_scores"]["ego_free_development"] -= 0.2
                gurbani_report["concerns"].append("Code may contain hardcoded credentials, potentially violating Ego-Free Development")
                gurbani_report["recommendations"].append("Use environment variables or secure storage for credentials")
            
            # Universal Design: Check for accessibility considerations
            if "color" in code.lower() and "contrast" not in code.lower():
                gurbani_report["principle_scores"]["universal_design"] -= 0.1
                gurbani_report["concerns"].append("Code may not consider accessibility, potentially violating Universal Design")
                gurbani_report["recommendations"].append("Ensure UI elements have proper contrast and accessibility features")
            
            # Mindful Resource Usage: Check for resource management
            if "open(" in code and "close(" not in code:
                gurbani_report["principle_scores"]["mindful_resource_usage"] -= 0.2
                gurbani_report["concerns"].append("Code may not properly close resources, violating Mindful Resource Usage")
                gurbani_report["recommendations"].append("Use context managers (with statements) for resource management")
            
            # Recalculate overall score based on principle scores
            principle_scores = list(gurbani_report["principle_scores"].values())
            gurbani_report["score"] = round(sum(principle_scores) / len(principle_scores), 2)
            
            # Add general recommendations
            if gurbani_report["score"] < 0.7:
                gurbani_report["recommendations"].append("Review code for better alignment with Gurbani principles")
            if gurbani_report["score"] >= 0.9:
                gurbani_report["recommendations"].append("Code shows strong alignment with Gurbani principles")
            
            logger.info(f"Local Gurbani alignment evaluation completed with score: {gurbani_report['score']}")
            return gurbani_report
            
        except Exception as e:
            error_msg = f"Error during local Gurbani alignment evaluation: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
    
    async def get_improvement_suggestions(self, code: str, ethics_report: Dict[str, Any]) -> List[str]:
        """
        Get suggestions for improving code ethics using local heuristics.
        
        Args:
            code: The source code to evaluate
            ethics_report: The ethics report containing issues to address
            
        Returns:
            List of improvement suggestions
            
        Raises:
            LLMClientError: If generating suggestions fails
        """
        try:
            suggestions = []
            
            # Extract issues from the ethics report
            issues = ethics_report.get("potential_issues", [])
            
            # Generate suggestions based on issues
            for issue in issues:
                category = issue.get("category", "")
                description = issue.get("description", "")
                
                if "system calls" in description:
                    suggestions.append("Replace system calls with safer alternatives like subprocess module with shell=False")
                
                if "network operations" in description:
                    suggestions.append("Add privacy notices and user consent before performing network operations")
                    suggestions.append("Implement data minimization principles for network requests")
                
                if "file operations" in description:
                    suggestions.append("Restrict file operations to specific directories using os.path.abspath")
                    suggestions.append("Validate file paths before operations to prevent path traversal attacks")
                
                if "harmful term" in description:
                    term = description.split(":")[-1].strip()
                    suggestions.append(f"Rename functions containing '{term}' to use more neutral terminology")
            
            # Add general suggestions based on category scores
            for category, data in ethics_report.get("categories", {}).items():
                score = data.get("score", 1.0)
                
                if category == "privacy" and score < 0.8:
                    suggestions.append("Implement data minimization principles")
                    suggestions.append("Add clear privacy notices and user consent mechanisms")
                
                if category == "security" and score < 0.8:
                    suggestions.append("Validate all inputs to prevent injection attacks")
                    suggestions.append("Use principle of least privilege for all operations")
                
                if category == "fairness" and score < 0.8:
                    suggestions.append("Review code for potential bias in algorithms or data processing")
                    suggestions.append("Ensure equal treatment of all user groups")
                
                if category == "transparency" and score < 0.8:
                    suggestions.append("Add comprehensive documentation explaining code behavior")
                    suggestions.append("Implement logging to make operations traceable")
                
                if category == "accountability" and score < 0.8:
                    suggestions.append("Add audit trails for sensitive operations")
                    suggestions.append("Implement proper error handling with meaningful messages")
                
                if category == "gurbani_alignment" and score < 0.8:
                    suggestions.append("Review code against Gurbani principles, especially regarding service to users")
                    suggestions.append("Ensure code promotes unity and balance rather than division")
            
            # Deduplicate suggestions
            suggestions = list(set(suggestions))
            
            # If no specific suggestions, add general ones
            if not suggestions:
                suggestions = [
                    "Add comprehensive documentation to improve transparency",
                    "Implement proper error handling with user-friendly messages",
                    "Review code for potential security vulnerabilities",
                    "Consider privacy implications of data handling",
                    "Ensure code is accessible and inclusive for all users"
                ]
            
            logger.info(f"Generated {len(suggestions)} local improvement suggestions")
            return suggestions
            
        except Exception as e:
            error_msg = f"Error generating local improvement suggestions: {str(e)}"
            logger.error(error_msg)
            raise LLMClientError(error_msg)
    
    async def health_check(self) -> bool:
        """
        Check if the local model is available.
        
        Returns:
            Always returns True since local model is always available
        """
        return True