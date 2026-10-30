"""Pydantic v2 request and response DTOs standardized in English."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

MIN_TEXT_BATCH_SIZE = 1
MAX_TEXT_BATCH_SIZE = 50


class ClassificationRequest(BaseModel):
    """Input payload for multi-label card description classification."""

    texts: List[str] = Field(
        ...,
        description="List of card effect descriptions to classify",
        min_length=MIN_TEXT_BATCH_SIZE,
        max_length=MAX_TEXT_BATCH_SIZE,
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Custom probability threshold (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    @field_validator("texts")
    @classmethod
    def validate_non_empty_texts(cls, values: List[str]) -> List[str]:
        cleaned = [item.strip() for item in values if item and item.strip()]
        if not cleaned:
            raise ValueError("texts must contain at least one non-empty description")
        return cleaned


class GenrePrediction(BaseModel):
    """Predicted probability and confidence level for a single card genre."""

    genre: str = Field(..., description="Genre name")
    probability: float = Field(..., description="Sigmoid probability (0.0-1.0)", ge=0.0, le=1.0)
    confidence: str = Field(..., description="Confidence tier (High, Medium, Low)")


class SingleClassificationResult(BaseModel):
    """Classification output for a single card description."""

    text: str = Field(..., description="Original card description")
    detected_genres: List[GenrePrediction] = Field(default_factory=list)
    primary_genres: List[str] = Field(default_factory=list)
    processing_time: float = Field(..., ge=0.0)


class ClassificationResponse(BaseModel):
    """Batch classification response with statistics and model metadata."""

    results: List[SingleClassificationResult]
    statistics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class HealthResponse(BaseModel):
    """Service readiness and hardware status response."""

    status: str
    version: str
    model_loaded: bool
    device: str
    gpu_memory: Optional[Dict[str, str]] = None
    available_genres: List[str]


class GenreListResponse(BaseModel):
    """Response listing all supported card genres."""

    genres: List[str]
    total: int
    description: str = "Supported Yu-Gi-Oh! card mechanics genres"


class ErrorResponse(BaseModel):
    """Standardized HTTP error payload."""

    error: str
    message: str
    status_code: int
    details: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
