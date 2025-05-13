"""
API routes for the LLM Ethics Service.

This module defines the FastAPI routes for the LLM ethics service,
providing endpoints for code evaluation, health checks, and other
service functionality.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path, Body
from fastapi.responses import JSONResponse

from ..config import settings
from ..models.llm_client import LLMClient, LLMClientError
from ..models.claude_client import ClaudeClient
from ..models.gpt_client import GPTClient
from ..models.local_model_client import LocalModelClient
from ..services.ethics_evaluator import EthicsEvaluator, EthicsEvaluatorError
from .schemas import (
    CodeEvaluationRequest,
    ComprehensiveEvaluationResponse,
    EthicsEvaluationResponse,
    GurbaniAlignmentResponse,
    HealthCheckResponse,
    ErrorResponse,
    LLMProvider
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix=settings.API_PREFIX)

# LLM client instances
llm_clients = {
    LLMProvider.CLAUDE: None,
    LLMProvider.GPT: None,
    LLMProvider.LOCAL: LocalModelClient()
}

# Ethics evaluator instances
ethics_evaluators = {}

def get_llm_client(provider: Optional[LLMProvider] = None) -> LLMClient:
    """
    Get an LLM client instance.
    
    Args:
        provider: LLM provider to use (defaults to configured default)
        
    Returns:
        LLM client instance
        
    Raises:
        HTTPException: If the requested provider is not available
    """
    # Use default provider if not specified
    if provider is None:
        provider = LLMProvider(settings.DEFAULT_LLM_PROVIDER)
    
    # Initialize client if not already initialized
    if provider == LLMProvider.CLAUDE and llm_clients[provider] is None:
        if not settings.CLAUDE_API_KEY:
            raise HTTPException(
                status_code=503,
                detail=f"Claude API key not configured. Please set CLAUDE_API_KEY environment variable."
            )
        llm_clients[provider] = ClaudeClient(api_key=settings.CLAUDE_API_KEY, model=settings.CLAUDE_MODEL)
    
    elif provider == LLMProvider.GPT and llm_clients[provider] is None:
        if not settings.OPENAI_API_KEY:
            raise HTTPException(
                status_code=503,
                detail=f"OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
            )
        llm_clients[provider] = GPTClient(api_key=settings.OPENAI_API_KEY, model=settings.GPT_MODEL)
    
    # Return the client
    client = llm_clients[provider]
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=f"LLM provider {provider} is not available"
        )
    
    return client

def get_ethics_evaluator(provider: Optional[LLMProvider] = None) -> EthicsEvaluator:
    """
    Get an ethics evaluator instance.
    
    Args:
        provider: LLM provider to use (defaults to configured default)
        
    Returns:
        Ethics evaluator instance
    """
    # Use default provider if not specified
    if provider is None:
        provider = LLMProvider(settings.DEFAULT_LLM_PROVIDER)
    
    # Initialize evaluator if not already initialized
    if provider not in ethics_evaluators:
        client = get_llm_client(provider)
        ethics_evaluators[provider] = EthicsEvaluator(client)
    
    return ethics_evaluators[provider]

@router.post(
    "/evaluate",
    response_model=ComprehensiveEvaluationResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    summary="Evaluate code ethics",
    description="Evaluate the ethics of code using LLM analysis"
)
async def evaluate_code(request: CodeEvaluationRequest):
    """
    Evaluate the ethics of code.
    
    This endpoint performs a comprehensive ethics evaluation of the provided code,
    optionally including Gurbani alignment and improvement suggestions.
    
    Args:
        request: Code evaluation request
        
    Returns:
        Comprehensive evaluation response
    """
    try:
        # Get ethics evaluator
        evaluator = get_ethics_evaluator(request.provider)
        
        # Perform ethics evaluation
        ethics_report = await evaluator.evaluate_ethics(request.code)
        
        # Initialize response
        response = {
            "ethics_evaluation": ethics_report,
            "gurbani_alignment": None,
            "improvement_suggestions": None,
            "overall_assessment": {
                "ethics_score": ethics_report.get("overall_score", 0.0),
                "gurbani_score": None,
                "combined_score": ethics_report.get("overall_score", 0.0),
                "passed": ethics_report.get("overall_score", 0.0) >= settings.ETHICS_PASSING_THRESHOLD
            },
            "timestamp": time.time()
        }
        
        # Add Gurbani alignment if requested
        if request.include_gurbani_alignment:
            gurbani_report = await evaluator.evaluate_gurbani_alignment(request.code, ethics_report)
            response["gurbani_alignment"] = gurbani_report
            response["overall_assessment"]["gurbani_score"] = gurbani_report.get("score", 0.0)
            
            # Update combined score and pass status
            ethics_score = ethics_report.get("overall_score", 0.0)
            gurbani_score = gurbani_report.get("score", 0.0)
            response["overall_assessment"]["combined_score"] = (ethics_score + gurbani_score) / 2
            response["overall_assessment"]["passed"] = (
                ethics_score >= settings.ETHICS_PASSING_THRESHOLD and
                gurbani_score >= settings.GURBANI_ALIGNMENT_THRESHOLD
            )
        
        # Add improvement suggestions if requested
        if request.include_suggestions:
            suggestions = await evaluator.get_improvement_suggestions(request.code, ethics_report)
            response["improvement_suggestions"] = suggestions
        
        logger.info(f"Completed code evaluation with combined score: {response['overall_assessment']['combined_score']}")
        return response
        
    except EthicsEvaluatorError as e:
        logger.error(f"Ethics evaluation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Ethics evaluation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@router.post(
    "/evaluate/ethics",
    response_model=EthicsEvaluationResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    summary="Evaluate code ethics only",
    description="Evaluate only the ethics of code without Gurbani alignment or suggestions"
)
async def evaluate_ethics(
    code: str = Body(..., description="Source code to evaluate"),
    provider: Optional[LLMProvider] = Query(None, description="LLM provider to use")
):
    """
    Evaluate only the ethics of code.
    
    This endpoint performs an ethics evaluation of the provided code
    without Gurbani alignment or improvement suggestions.
    
    Args:
        code: Source code to evaluate
        provider: LLM provider to use (defaults to configured default)
        
    Returns:
        Ethics evaluation response
    """
    try:
        # Get ethics evaluator
        evaluator = get_ethics_evaluator(provider)
        
        # Perform ethics evaluation
        ethics_report = await evaluator.evaluate_ethics(code)
        
        logger.info(f"Completed ethics evaluation with score: {ethics_report.get('overall_score', 'N/A')}")
        return ethics_report
        
    except EthicsEvaluatorError as e:
        logger.error(f"Ethics evaluation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Ethics evaluation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@router.post(
    "/evaluate/gurbani",
    response_model=GurbaniAlignmentResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    summary="Evaluate Gurbani alignment only",
    description="Evaluate only the Gurbani alignment of code"
)
async def evaluate_gurbani_alignment(
    code: str = Body(..., description="Source code to evaluate"),
    provider: Optional[LLMProvider] = Query(None, description="LLM provider to use")
):
    """
    Evaluate only the Gurbani alignment of code.
    
    This endpoint performs a Gurbani alignment evaluation of the provided code.
    
    Args:
        code: Source code to evaluate
        provider: LLM provider to use (defaults to configured default)
        
    Returns:
        Gurbani alignment response
    """
    try:
        # Get ethics evaluator
        evaluator = get_ethics_evaluator(provider)
        
        # First perform ethics evaluation (required for context)
        ethics_report = await evaluator.evaluate_ethics(code)
        
        # Then evaluate Gurbani alignment
        gurbani_report = await evaluator.evaluate_gurbani_alignment(code, ethics_report)
        
        logger.info(f"Completed Gurbani alignment evaluation with score: {gurbani_report.get('score', 'N/A')}")
        return gurbani_report
        
    except EthicsEvaluatorError as e:
        logger.error(f"Gurbani alignment evaluation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Gurbani alignment evaluation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@router.post(
    "/suggestions",
    response_model=List[str],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    },
    summary="Get improvement suggestions",
    description="Get suggestions for improving code ethics"
)
async def get_suggestions(
    code: str = Body(..., description="Source code to evaluate"),
    provider: Optional[LLMProvider] = Query(None, description="LLM provider to use")
):
    """
    Get suggestions for improving code ethics.
    
    This endpoint provides suggestions for improving the ethics of the provided code.
    
    Args:
        code: Source code to evaluate
        provider: LLM provider to use (defaults to configured default)
        
    Returns:
        List of improvement suggestions
    """
    try:
        # Get ethics evaluator
        evaluator = get_ethics_evaluator(provider)
        
        # First perform ethics evaluation (required for context)
        ethics_report = await evaluator.evaluate_ethics(code)
        
        # Then get improvement suggestions
        suggestions = await evaluator.get_improvement_suggestions(code, ethics_report)
        
        logger.info(f"Generated {len(suggestions)} improvement suggestions")
        return suggestions
        
    except EthicsEvaluatorError as e:
        logger.error(f"Suggestion generation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Suggestion generation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during suggestion generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Check service health",
    description="Check the health of the LLM ethics service"
)
async def health_check():
    """
    Check the health of the LLM ethics service.
    
    This endpoint checks the health of the service, including the availability
    of LLM providers and cache statistics.
    
    Returns:
        Health check response
    """
    try:
        # Check LLM provider availability
        provider_status = {}
        
        # Check Claude availability
        try:
            if settings.CLAUDE_API_KEY:
                claude_client = get_llm_client(LLMProvider.CLAUDE)
                provider_status[LLMProvider.CLAUDE] = await claude_client.health_check()
            else:
                provider_status[LLMProvider.CLAUDE] = False
        except Exception:
            provider_status[LLMProvider.CLAUDE] = False
        
        # Check GPT availability
        try:
            if settings.OPENAI_API_KEY:
                gpt_client = get_llm_client(LLMProvider.GPT)
                provider_status[LLMProvider.GPT] = await gpt_client.health_check()
            else:
                provider_status[LLMProvider.GPT] = False
        except Exception:
            provider_status[LLMProvider.GPT] = False
        
        # Local model is always available
        provider_status[LLMProvider.LOCAL] = True
        
        # Get cache statistics
        cache_stats = {}
        if LLMProvider.CLAUDE in ethics_evaluators:
            cache_stats = ethics_evaluators[LLMProvider.CLAUDE].get_cache_stats()
        elif LLMProvider.GPT in ethics_evaluators:
            cache_stats = ethics_evaluators[LLMProvider.GPT].get_cache_stats()
        elif LLMProvider.LOCAL in ethics_evaluators:
            cache_stats = ethics_evaluators[LLMProvider.LOCAL].get_cache_stats()
        else:
            # Create a temporary evaluator to get cache stats
            temp_evaluator = get_ethics_evaluator(LLMProvider.LOCAL)
            cache_stats = temp_evaluator.get_cache_stats()
        
        # Determine overall status
        default_provider = LLMProvider(settings.DEFAULT_LLM_PROVIDER)
        if provider_status.get(default_provider, False):
            status = "ok"
        elif any(provider_status.values()):
            status = "degraded"
        else:
            status = "error"
        
        return {
            "status": status,
            "version": settings.VERSION,
            "llm_providers": provider_status,
            "cache": cache_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "error",
            "version": settings.VERSION,
            "llm_providers": {},
            "cache": {},
            "timestamp": time.time()
        }

@router.post(
    "/cache/clear",
    summary="Clear evaluation cache",
    description="Clear the evaluation cache for all providers"
)
async def clear_cache():
    """
    Clear the evaluation cache for all providers.
    
    This endpoint clears the evaluation cache for all LLM providers.
    
    Returns:
        Success message
    """
    try:
        # Clear cache for all evaluators
        for evaluator in ethics_evaluators.values():
            evaluator.clear_cache()
        
        logger.info("Cleared evaluation cache for all providers")
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Cache clear error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )