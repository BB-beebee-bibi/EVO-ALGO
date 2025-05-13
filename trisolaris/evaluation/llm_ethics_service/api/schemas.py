"""
API schemas for the LLM Ethics Service.

This module defines the Pydantic models used for request and response validation
in the LLM ethics service API. These schemas ensure proper data structure and
validation for all API endpoints.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    CLAUDE = "claude"
    GPT = "gpt"
    LOCAL = "local"

class CodeEvaluationRequest(BaseModel):
    """
    Request schema for code evaluation.
    
    This model defines the structure for requests to evaluate code ethics.
    """
    code: str = Field(..., description="Source code to evaluate")
    provider: Optional[LLMProvider] = Field(
        None, 
        description="LLM provider to use for evaluation (defaults to configured default)"
    )
    include_gurbani_alignment: bool = Field(
        True, 
        description="Whether to include Gurbani principles alignment in the evaluation"
    )
    include_suggestions: bool = Field(
        True, 
        description="Whether to include improvement suggestions in the evaluation"
    )
    
    @validator('code')
    def code_must_not_be_empty(cls, v):
        """Validate that code is not empty."""
        if not v or not v.strip():
            raise ValueError("Code cannot be empty")
        return v

class CategoryScore(BaseModel):
    """
    Schema for category-specific ethics scores and concerns.
    
    This model defines the structure for ethics scores and concerns
    for a specific category.
    """
    score: float = Field(..., description="Score from 0.0 to 1.0")
    concerns: List[str] = Field(default_factory=list, description="List of concerns for this category")
    recommendations: List[str] = Field(default_factory=list, description="List of recommendations for this category")

class PotentialIssue(BaseModel):
    """
    Schema for potential ethical issues.
    
    This model defines the structure for potential ethical issues
    identified in the code.
    """
    category: str = Field(..., description="Category of the issue (e.g., security, privacy)")
    severity: str = Field(..., description="Severity of the issue (high, medium, low)")
    description: str = Field(..., description="Description of the issue")

class EthicsEvaluationResponse(BaseModel):
    """
    Response schema for ethics evaluation.
    
    This model defines the structure for ethics evaluation responses.
    """
    evaluation_id: str = Field(..., description="Unique identifier for this evaluation")
    timestamp: str = Field(..., description="Timestamp of the evaluation")
    overall_score: float = Field(..., description="Overall ethics score from 0.0 to 1.0")
    categories: Dict[str, CategoryScore] = Field(..., description="Scores and concerns by category")
    potential_issues: List[PotentialIssue] = Field(default_factory=list, description="List of potential ethical issues")
    recommendations: List[str] = Field(default_factory=list, description="List of general recommendations")

class PrincipleScore(BaseModel):
    """
    Schema for Gurbani principle scores.
    
    This model defines the structure for scores for each Gurbani principle.
    """
    unity_in_design: float = Field(..., description="Unity in Design principle score")
    natural_flow: float = Field(..., description="Natural Flow principle score")
    truth_and_transparency: float = Field(..., description="Truth and Transparency principle score")
    service_oriented: float = Field(..., description="Service-Oriented Architecture principle score")
    balance_and_harmony: float = Field(..., description="Balance and Harmony principle score")
    ego_free_development: float = Field(..., description="Ego-Free Development principle score")
    universal_design: float = Field(..., description="Universal Design principle score")
    mindful_resource_usage: float = Field(..., description="Mindful Resource Usage principle score")

class GurbaniAlignmentResponse(BaseModel):
    """
    Response schema for Gurbani alignment evaluation.
    
    This model defines the structure for Gurbani alignment evaluation responses.
    """
    score: float = Field(..., description="Overall Gurbani alignment score from 0.0 to 1.0")
    principle_scores: PrincipleScore = Field(..., description="Scores for each Gurbani principle")
    concerns: List[str] = Field(default_factory=list, description="List of Gurbani alignment concerns")
    recommendations: List[str] = Field(default_factory=list, description="List of recommendations for better alignment")

class ComprehensiveEvaluationResponse(BaseModel):
    """
    Response schema for comprehensive code evaluation.
    
    This model defines the structure for comprehensive evaluation responses,
    including ethics evaluation, Gurbani alignment, and improvement suggestions.
    """
    ethics_evaluation: EthicsEvaluationResponse = Field(..., description="Ethics evaluation results")
    gurbani_alignment: Optional[GurbaniAlignmentResponse] = Field(None, description="Gurbani alignment results")
    improvement_suggestions: Optional[List[str]] = Field(None, description="List of improvement suggestions")
    overall_assessment: Dict[str, Any] = Field(..., description="Overall assessment summary")
    timestamp: float = Field(..., description="Timestamp of the evaluation")

class HealthCheckResponse(BaseModel):
    """
    Response schema for service health check.
    
    This model defines the structure for health check responses.
    """
    status: str = Field(..., description="Service status (ok, degraded, error)")
    version: str = Field(..., description="Service version")
    llm_providers: Dict[str, bool] = Field(..., description="Status of each LLM provider")
    cache: Dict[str, Any] = Field(..., description="Cache statistics")
    timestamp: float = Field(..., description="Timestamp of the health check")

class ErrorResponse(BaseModel):
    """
    Response schema for API errors.
    
    This model defines the structure for error responses.
    """
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")
    timestamp: float = Field(..., description="Timestamp of the error")