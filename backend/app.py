from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from recommendation import CropRecommendationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

service: CropRecommendationService | None = None


class RecommendationRequest(BaseModel):
    rainfall: float = Field(ge=0, description="Annual rainfall in millimetres")
    expected_yield: float = Field(ge=0, description="Expected crop yield")
    market_price: float = Field(ge=0, description="Expected market price")
    savings: float = Field(ge=0, description="Available investment or savings")
    previous_crop: str = Field(default="None")
    current_year: int = Field(default=0, ge=0, le=29)


class RecommendationResponse(BaseModel):
    recommended_crop: str
    action_index: int
    estimated_profit: float
    rotation_penalty: float
    annual_living_cost: float
    estimated_net_income: float
    updated_savings: float
    explanation: str
    closest_context: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service

    logger.info("Loading PPO crop recommendation service...")
    service = CropRecommendationService()
    logger.info("PPO crop recommendation service loaded successfully.")

    yield

    service = None


app = FastAPI(
    title="AgriPlan AI API",
    version="1.0.0",
    description="FastAPI backend for the PPO Farmer Crop and Income Planning project.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to your deployed frontend URL in production.
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "name": "AgriPlan AI API",
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "model_loaded": service is not None,
    }


@app.get("/crops")
def crops() -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail="Model service is not ready.")

    return {"crops": service.crops}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail="Model service is not ready.")

    try:
        return service.recommend(
            rainfall=request.rainfall,
            expected_yield=request.expected_yield,
            market_price=request.market_price,
            savings=request.savings,
            previous_crop=request.previous_crop,
            current_year=request.current_year,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Recommendation failed.")
        raise HTTPException(
            status_code=500,
            detail="The PPO recommendation could not be generated.",
        ) from exc
