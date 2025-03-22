"""
FastAPI Quickstart with Instructor

This example demonstrates how to:
1. Create a FastAPI application with structured LLM outputs
2. Use Instructor for type-safe LLM responses
3. Integrate with our configuration
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import instructor
from openai import OpenAI
import uvicorn
from enum import Enum

# Import our settings
from e1_pydantic_settings import get_settings

# Initialize FastAPI app
app = FastAPI(title="AI Service API")

# Get settings
settings = get_settings()

# Initialize OpenAI client with Instructor
client = instructor.patch(OpenAI(api_key=settings.openai_api_key))


# Define sentiment enum
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# Define structured output model
class Analysis(BaseModel):
    """Structured output for text analysis"""

    summary: str = Field(description="Brief summary of the input text")
    sentiment: Sentiment = Field(
        description="Sentiment analysis of the text (positive/negative/neutral)"
    )
    key_points: list[str] = Field(description="Main points from the text")


# Define request model
class AIRequest(BaseModel):
    """Input data model"""

    text: str
    max_tokens: Optional[int] = 1000


@app.get("/")
def read_root():
    """Health check endpoint"""
    return {"status": "ok", "model": settings.openai_model}


@app.post("/analyze", response_model=Analysis)
async def analyze_text(request: AIRequest):
    """Analyze text using OpenAI with structured output"""
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            response_model=Analysis,
            messages=[
                {"role": "user", "content": f"Analyze this text: {request.text}"}
            ],
            max_tokens=request.max_tokens,
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\nStarting FastAPI server...")
    print("Run e2_test_fastapi.py in another terminal to test the endpoints")
    print("Or visit http://localhost:8000/docs for interactive documentation")

    uvicorn.run(app, host="0.0.0.0", port=8000)
