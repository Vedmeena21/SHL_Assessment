"""
FastAPI REST API for SHL Assessment Recommendations

Endpoints:
- GET /health - Health check
- POST /recommend - Get assessment recommendations
"""

import logging
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from embedding.vectorstore import AssessmentVectorStore, load_assessments_from_file
from recommender.engine import RecommendationEngine

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommender API",
    description="Intelligent recommendation system for SHL assessments",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[RecommendationEngine] = None


# Pydantic models
class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="API status")
    message: str = Field(..., description="Status message")
    assessments_count: Optional[int] = Field(None, description="Number of assessments indexed")


class RecommendRequest(BaseModel):
    """Request model for recommendations"""
    input_text: str = Field(..., description="Job description or search query", min_length=10)
    input_type: Optional[str] = Field("query", description="Type of input: 'query' or 'jd'")
    max_recommendations: Optional[int] = Field(10, description="Maximum number of recommendations", ge=1, le=20)


class Assessment(BaseModel):
    """Assessment model"""
    assessment_name: str = Field(..., description="Name of the assessment")
    assessment_url: str = Field(..., description="URL to the assessment")
    relevance_score: Optional[float] = Field(None, description="Relevance score (0-1)")
    test_type: Optional[str] = Field(None, description="Test type (K=Knowledge, P=Personality, O=Other)")


class RecommendResponse(BaseModel):
    """Response model for recommendations"""
    query: str = Field(..., description="Original query")
    recommendations: List[Assessment] = Field(..., description="List of recommended assessments")
    total_recommendations: int = Field(..., description="Total number of recommendations returned")


@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation engine on startup"""
    global engine
    
    try:
        logger.info("Initializing recommendation engine...")
        
        # Initialize vector store
        vector_store = AssessmentVectorStore()
        
        # Load and index assessments if needed
        if vector_store.get_count() == 0:
            logger.info("Vector store is empty, loading assessments...")
            
            # Try to load from scraped data
            assessments_path = 'data/assessments.json'
            if os.path.exists(assessments_path):
                assessments = load_assessments_from_file(assessments_path)
                vector_store.index_assessments(assessments)
                logger.info(f"Indexed {len(assessments)} assessments")
            else:
                # Fallback: create sample data from CSV
                logger.warning(f"Assessments file not found at {assessments_path}")
                logger.warning("Using training data as fallback...")
                
                # Load from CSV and create basic assessment objects
                import pandas as pd
                df = pd.read_csv('Gen_AI Dataset.csv')
                
                logger.info(f"CSV columns: {df.columns.tolist()}")
                
                # Create unique assessments from URLs
                unique_assessments = {}
                for _, row in df.iterrows():
                    url = row['Assessment_url']
                    if url not in unique_assessments:
                        # Extract name from URL (last part before trailing slash)
                        url_parts = url.rstrip('/').split('/')
                        slug = url_parts[-1] if url_parts else 'unknown'
                        # Convert slug to readable name
                        name = slug.replace('-new', '').replace('-', ' ').strip().title()
                        
                        # Clean up common patterns
                        if not name or name == 'View':
                            name = slug
                        
                        unique_assessments[url] = {
                            'name': name,
                            'url': url,
                            'test_type': 'K',  # Default to Knowledge test
                            'description': f"SHL assessment: {name}",
                            'categories': ['assessment'],
                            'full_content': f"{name} SHL assessment"
                        }
                
                assessments = list(unique_assessments.values())
                vector_store.index_assessments(assessments)
                logger.info(f"Indexed {len(assessments)} assessments from CSV")
        else:
            logger.info(f"Vector store already contains {vector_store.get_count()} assessments")
        
        # Initialize recommendation engine
        engine = RecommendationEngine(vector_store)
        logger.info("Recommendation engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns API status and basic information
    """
    try:
        assessments_count = engine.vector_store.get_count() if engine else 0
        
        return HealthResponse(
            status="healthy",
            message="SHL Assessment Recommender API is running",
            assessments_count=assessments_count
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            message=f"Error: {str(e)}"
        )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend_assessments(request: RecommendRequest):
    """
    Get assessment recommendations based on job description or query
    
    Args:
        request: Recommendation request with input text
        
    Returns:
        List of recommended assessments with relevance scores
    """
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine not initialized"
        )
    
    try:
        logger.info(f"Processing recommendation request: {request.input_text[:100]}...")
        
        # Get recommendations
        results = engine.recommend(
            query=request.input_text,
            min_results=5,
            max_results=request.max_recommendations
        )
        
        # Format response
        assessments = []
        for result in results:
            assessments.append(Assessment(
                assessment_name=result['name'],
                assessment_url=result['url'],
                relevance_score=result.get('final_score'),
                test_type=result.get('test_type')
            ))
        
        response = RecommendResponse(
            query=request.input_text,
            recommendations=assessments,
            total_recommendations=len(assessments)
        )
        
        logger.info(f"Returned {len(assessments)} recommendations")
        return response
        
    except Exception as e:
        logger.error(f"Error in recommendation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SHL Assessment Recommender API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend (POST)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    # Run server
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
