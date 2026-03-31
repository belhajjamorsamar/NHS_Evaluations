"""
FastAPI REST API for the ShopVite FAQ Assistant.
Exposes /ask and /health endpoints with full error handling.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import traceback

from src.config import config
from src.retrieval import RAGPipeline
from src.logger import logger

# Initialize FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="RAG-based FAQ Assistant for ShopVite E-Commerce",
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline (global state)
rag_pipeline: Optional[RAGPipeline] = None


# Request/Response models
class QuestionRequest(BaseModel):
    """Request model for /ask endpoint."""

    question: str = Field(..., min_length=1, max_length=1000)
    k: Optional[int] = Field(default=3, ge=1, le=10, description="Number of documents to retrieve")


class AnswerResponse(BaseModel):
    """Response model for successful queries."""

    answer: str
    sources: List[str]
    confidence: str = Field(..., pattern="^(high|medium|low)$")
    context_used: int
    out_of_scope: bool = False
    query: str
    model: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: str
    status_code: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., pattern="^(healthy|unhealthy)$")
    pipeline_initialized: bool
    vector_store_ready: bool
    llm_model: str
    embedding_model: str
    timestamp: str


# Startup/Shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on application startup."""
    global rag_pipeline

    try:
        logger.info("Starting up ShopVite FAQ Assistant API...")
        rag_pipeline = RAGPipeline()

        # Try loading existing vector store
        vector_store_exists = False
        try:
            rag_pipeline.load_existing()
            # Check if vector store actually has documents
            if rag_pipeline.vector_store.vector_store is not None:
                collection = rag_pipeline.vector_store.vector_store._collection
                doc_count = collection.count() if hasattr(collection, 'count') else 0
                if doc_count > 0:
                    vector_store_exists = True
                    logger.info(f"Loaded existing vector store with {doc_count} documents")
        except Exception as e:
            logger.info(f"Could not load existing vector store: {str(e)}")

        # If no valid vector store exists, initialize fresh
        if not vector_store_exists:
            logger.info("Initializing new vector store from documents...")
            rag_pipeline.initialize(config.DATA_DIR)
            logger.info("Vector store initialized successfully")

        logger.info("API startup completed successfully")

    except Exception as e:
        logger.error(f"Failed to initialize pipeline on startup: {str(e)}")
        logger.error(traceback.format_exc())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down ShopVite FAQ Assistant API...")


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse with system status
    """
    try:
        if not rag_pipeline or not rag_pipeline.is_initialized:
            logger.warning("Health check: pipeline not initialized")
            raise HTTPException(
                status_code=503,
                detail="Pipeline not initialized",
            )

        status_info = rag_pipeline.get_health_status()

        return HealthResponse(
            status="healthy" if status_info["pipeline_initialized"] else "unhealthy",
            pipeline_initialized=status_info["pipeline_initialized"],
            vector_store_ready=status_info["vector_store_ready"],
            llm_model=status_info["llm_model"],
            embedding_model=status_info["embedding_model"],
            timestamp=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during health check",
        )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest) -> AnswerResponse:
    """
    Main FAQ endpoint - answer questions based on RAG pipeline.

    Args:
        request: QuestionRequest with question and optional k parameter

    Returns:
        AnswerResponse with answer, sources, and confidence

    Raises:
        HTTPException: If pipeline not ready or query fails
    """
    try:
        # Validate pipeline
        if not rag_pipeline or not rag_pipeline.is_initialized:
            logger.error("Query received but pipeline not initialized")
            raise HTTPException(
                status_code=503,
                detail="FAQ Assistant is temporarily unavailable. Please try again later.",
            )

        logger.info(f"Received question: '{request.question}'")

        # Process query
        result = rag_pipeline.query(request.question, k=request.k)

        # Build response
        response = AnswerResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            confidence=result.get("confidence", "low"),
            context_used=result.get("context_used", 0),
            out_of_scope=result.get("out_of_scope", False),
            query=request.question,
            model=result.get("model", config.OPENAI_MODEL),
            timestamp=datetime.utcnow().isoformat(),
        )

        logger.info(
            f"Question processed successfully - "
            f"Confidence: {response.confidence}, "
            f"Out-of-scope: {response.out_of_scope}"
        )

        return response

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid question: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="FAQ Assistant encountered an error. Please try again later.",
        )
    except Exception as e:
        logger.error(f"Unexpected error in /ask endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing your question",
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": config.APP_NAME,
        "version": config.APP_VERSION,
        "description": "RAG-based FAQ Assistant for ShopVite E-Commerce",
        "endpoints": {
            "/health": "GET - Health check",
            "/ask": "POST - Ask a question",
            "/docs": "GET - API documentation",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())

    return {
        "error": "Internal Server Error",
        "detail": "An unexpected error occurred",
        "status_code": 500,
        "timestamp": datetime.utcnow().isoformat(),
    }


# Run with: uvicorn src.api:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG,
    )
