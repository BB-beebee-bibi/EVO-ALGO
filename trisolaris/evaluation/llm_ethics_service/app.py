"""
Main application module for the LLM Ethics Service.

This module initializes and configures the FastAPI application for the
LLM ethics service, setting up middleware, exception handlers, and routes.
"""

import time
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .config import settings
from .api.routes import router
from .api.schemas import ErrorResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="LLM Ethics Service",
    description="A service for evaluating code ethics using LLMs",
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request information and timing."""
    start_time = time.time()
    
    # Get client IP and request details
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    
    logger.info(f"Request started: {method} {path} from {client_ip}")
    
    # Process the request
    response = await call_next(request)
    
    # Calculate and log request duration
    duration = time.time() - start_time
    status_code = response.status_code
    logger.info(f"Request completed: {method} {path} - {status_code} in {duration:.3f}s")
    
    return response

# Add exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.detail),
            detail=None,
            code=f"HTTP_{exc.status_code}",
            timestamp=time.time()
        ).dict()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Validation error",
            detail=str(exc),
            code="VALIDATION_ERROR",
            timestamp=time.time()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            code="INTERNAL_ERROR",
            timestamp=time.time()
        ).dict()
    )

# Include API routes
app.include_router(router)

# Add root endpoint
@app.get("/")
async def root():
    """Root endpoint that redirects to documentation."""
    return {
        "service": "LLM Ethics Service",
        "version": settings.VERSION,
        "documentation": "/api/docs"
    }

# Add health check endpoint at root level
@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}

if __name__ == "__main__":
    # Run the application using uvicorn when executed directly
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )