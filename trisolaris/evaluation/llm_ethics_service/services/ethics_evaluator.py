"""
Ethics Evaluator Service for the LLM Ethics Service.

This module provides the core ethics evaluation functionality, coordinating
between different LLM clients and applying caching for performance. It serves
as the main service layer for the ethics evaluation system.
"""

import ast
import json
import logging
import time
import hashlib
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from functools import lru_cache

from ..config import settings
from ..models.llm_client import LLMClient, LLMClientError
from .gurbani_principles import GurbaniPrinciples, GurbaniPrinciplesError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EthicsEvaluatorError(Exception):
    """Exception raised when ethics evaluation fails."""
    pass

class EthicsEvaluator:
    """
    Service for evaluating code ethics using LLM clients.
    
    This class coordinates between different LLM clients, applies caching,
    and provides a unified interface for ethics evaluation.
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize the ethics evaluator service.
        
        Args:
            llm_client: LLM client to use for evaluation
        """
        self.llm_client = llm_client
        self.gurbani_principles = GurbaniPrinciples()
        self.cache = {}
        self.cache_ttl = settings.CACHE_TTL
        self.cache_max_size = settings.CACHE_MAX_SIZE
        self.cache_enabled = settings.CACHE_ENABLED
        logger.info(f"Initialized ethics evaluator with {llm_client.__class__.__name__}")
    
    def _get_cache_key(self, code: str) -> str:
        """
        Generate a cache key for a code string.
        
        Args:
            code: Source code string
            
        Returns:
            Cache key as a string
        """
        return hashlib.md5(code.encode()).hexdigest()
    
    def _get_from_cache(self, code: str) -> Optional[Dict[str, Any]]:
        """
        Get evaluation result from cache if available and not expired.
        
        Args:
            code: Source code to evaluate
            
        Returns:
            Cached evaluation result or None if not in cache or expired
        """
        if not self.cache_enabled:
            return None
        
        cache_key = self._get_cache_key(code)
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if time.time() - cached_result['timestamp'] < self.cache_ttl:
                logger.info("Using cached ethics evaluation result")
                return cached_result['result']
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        
        return None
    
    def _add_to_cache(self, code: str, result: Dict[str, Any]) -> None:
        """
        Add evaluation result to cache.
        
        Args:
            code: Source code that was evaluated
            result: Evaluation result to cache
        """
        if not self.cache_enabled:
            return
        
        # Manage cache size
        if len(self.cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        cache_key = self._get_cache_key(code)
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'result': result
        }
        logger.info("Added ethics evaluation result to cache")
    
    def _clean_cache(self) -> None:
        """Remove expired entries from cache."""
        if not self.cache_enabled:
            return
        
        current_time = time.time()
        expired_keys = [
            key for key, value in self.cache.items()
            if current_time - value['timestamp'] >= self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        logger.info(f"Cleaned {len(expired_keys)} expired entries from cache")
    
    async def evaluate_ethics(self, code: str) -> Dict[str, Any]:
        """
        Evaluate the ethics of code using the LLM client.
        
        Args:
            code: The source code to evaluate
            
        Returns:
            Dictionary containing the ethics evaluation report
            
        Raises:
            EthicsEvaluatorError: If the evaluation fails
        """
        try:
            # Check cache first
            cached_result = self._get_from_cache(code)
            if cached_result:
                return cached_result
            
            # Validate code
            if not code.strip():
                raise EthicsEvaluatorError("Cannot evaluate empty code")
            
            # Parse AST to validate syntax
            try:
                ast.parse(code)
            except SyntaxError as e:
                raise EthicsEvaluatorError(f"Cannot evaluate ethics due to syntax error: {str(e)}")
            except Exception as e:
                raise EthicsEvaluatorError(f"Error parsing code for ethics evaluation: {str(e)}")
            
            # Evaluate ethics using LLM client
            logger.info("Evaluating code ethics")
            ethics_report = await self.llm_client.evaluate_ethics(code)
            
            # Add to cache
            self._add_to_cache(code, ethics_report)
            
            logger.info(f"Ethics evaluation completed with score: {ethics_report.get('overall_score', 'N/A')}")
            return ethics_report
            
        except LLMClientError as e:
            error_msg = f"LLM client error during ethics evaluation: {str(e)}"
            logger.error(error_msg)
            raise EthicsEvaluatorError(error_msg)
        except Exception as e:
            error_msg = f"Error during ethics evaluation: {str(e)}"
            logger.error(error_msg)
            raise EthicsEvaluatorError(error_msg)
    
    async def evaluate_gurbani_alignment(self, code: str, ethics_report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate how well code aligns with Gurbani principles.
        
        Args:
            code: The source code to evaluate
            ethics_report: Optional ethics report for context (will be generated if not provided)
            
        Returns:
            Dictionary containing the Gurbani alignment evaluation
            
        Raises:
            EthicsEvaluatorError: If the evaluation fails
        """
        try:
            # Generate ethics report if not provided
            if ethics_report is None:
                ethics_report = await self.evaluate_ethics(code)
            
            # Check cache first
            cache_key = self._get_cache_key(code) + "_gurbani"
            if self.cache_enabled and cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if time.time() - cached_result['timestamp'] < self.cache_ttl:
                    logger.info("Using cached Gurbani alignment result")
                    return cached_result['result']
            
            # Evaluate Gurbani alignment using LLM client
            logger.info("Evaluating Gurbani alignment")
            gurbani_report = await self.llm_client.evaluate_gurbani_alignment(code, ethics_report)
            
            # Add to cache
            if self.cache_enabled:
                self.cache[cache_key] = {
                    'timestamp': time.time(),
                    'result': gurbani_report
                }
            
            logger.info(f"Gurbani alignment evaluation completed with score: {gurbani_report.get('score', 'N/A')}")
            return gurbani_report
            
        except LLMClientError as e:
            error_msg = f"LLM client error during Gurbani alignment evaluation: {str(e)}"
            logger.error(error_msg)
            raise EthicsEvaluatorError(error_msg)
        except Exception as e:
            error_msg = f"Error during Gurbani alignment evaluation: {str(e)}"
            logger.error(error_msg)
            raise EthicsEvaluatorError(error_msg)
    
    async def get_improvement_suggestions(self, code: str, ethics_report: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Get suggestions for improving code ethics.
        
        Args:
            code: The source code to evaluate
            ethics_report: Optional ethics report for context (will be generated if not provided)
            
        Returns:
            List of improvement suggestions
            
        Raises:
            EthicsEvaluatorError: If generating suggestions fails
        """
        try:
            # Generate ethics report if not provided
            if ethics_report is None:
                ethics_report = await self.evaluate_ethics(code)
            
            # Check cache first
            cache_key = self._get_cache_key(code) + "_suggestions"
            if self.cache_enabled and cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if time.time() - cached_result['timestamp'] < self.cache_ttl:
                    logger.info("Using cached improvement suggestions")
                    return cached_result['result']
            
            # Get improvement suggestions using LLM client
            logger.info("Generating improvement suggestions")
            suggestions = await self.llm_client.get_improvement_suggestions(code, ethics_report)
            
            # Add to cache
            if self.cache_enabled:
                self.cache[cache_key] = {
                    'timestamp': time.time(),
                    'result': suggestions
                }
            
            logger.info(f"Generated {len(suggestions)} improvement suggestions")
            return suggestions
            
        except LLMClientError as e:
            error_msg = f"LLM client error during suggestion generation: {str(e)}"
            logger.error(error_msg)
            raise EthicsEvaluatorError(error_msg)
        except Exception as e:
            error_msg = f"Error during suggestion generation: {str(e)}"
            logger.error(error_msg)
            raise EthicsEvaluatorError(error_msg)
    
    async def perform_comprehensive_evaluation(self, code: str) -> Dict[str, Any]:
        """
        Perform a comprehensive ethics evaluation including Gurbani alignment and suggestions.
        
        Args:
            code: The source code to evaluate
            
        Returns:
            Dictionary containing the comprehensive evaluation results
            
        Raises:
            EthicsEvaluatorError: If the evaluation fails
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(code) + "_comprehensive"
            if self.cache_enabled and cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if time.time() - cached_result['timestamp'] < self.cache_ttl:
                    logger.info("Using cached comprehensive evaluation result")
                    return cached_result['result']
            
            # Perform all evaluations
            ethics_report = await self.evaluate_ethics(code)
            gurbani_report = await self.evaluate_gurbani_alignment(code, ethics_report)
            suggestions = await self.get_improvement_suggestions(code, ethics_report)
            
            # Compile comprehensive report
            comprehensive_report = {
                "ethics_evaluation": ethics_report,
                "gurbani_alignment": gurbani_report,
                "improvement_suggestions": suggestions,
                "overall_assessment": {
                    "ethics_score": ethics_report.get("overall_score", 0.0),
                    "gurbani_score": gurbani_report.get("score", 0.0),
                    "combined_score": (ethics_report.get("overall_score", 0.0) + gurbani_report.get("score", 0.0)) / 2,
                    "passed": ethics_report.get("overall_score", 0.0) >= settings.ETHICS_PASSING_THRESHOLD and 
                             gurbani_report.get("score", 0.0) >= settings.GURBANI_ALIGNMENT_THRESHOLD
                },
                "timestamp": time.time()
            }
            
            # Add to cache
            if self.cache_enabled:
                self.cache[cache_key] = {
                    'timestamp': time.time(),
                    'result': comprehensive_report
                }
            
            logger.info(f"Comprehensive evaluation completed with combined score: {comprehensive_report['overall_assessment']['combined_score']}")
            return comprehensive_report
            
        except Exception as e:
            error_msg = f"Error during comprehensive evaluation: {str(e)}"
            logger.error(error_msg)
            raise EthicsEvaluatorError(error_msg)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary containing cache statistics
        """
        if not self.cache_enabled:
            return {"enabled": False, "size": 0, "max_size": self.cache_max_size, "ttl": self.cache_ttl}
        
        current_time = time.time()
        active_entries = sum(1 for value in self.cache.values() if current_time - value['timestamp'] < self.cache_ttl)
        
        return {
            "enabled": True,
            "size": len(self.cache),
            "active_entries": active_entries,
            "expired_entries": len(self.cache) - active_entries,
            "max_size": self.cache_max_size,
            "ttl": self.cache_ttl,
            "hit_ratio": 0.0  # Would need to track hits/misses to calculate this
        }
    
    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self.cache = {}
        logger.info("Cleared ethics evaluation cache")