"""FastAPI entrypoint for the Yu-Gi-Oh! Card Genre Classifier API."""
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    ErrorResponse,
    GenreListResponse,
    HealthResponse,
)
from app.services.classifier import (
    BatchStatisticsCalculator,
    CardGenreClassifier,
    GenreClassifierPort,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initializes and tears down the classifier service on app startup/shutdown."""
    classifier_service = CardGenreClassifier()
    await classifier_service.load_model()
    application.state.classifier = classifier_service
    yield
    await classifier_service.cleanup()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Multi-label Yu-Gi-Oh! card mechanics genre classifier powered by Transformers.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_classifier_service(request: Request) -> GenreClassifierPort:
    """Dependency provider returning the active GenreClassifierPort instance."""
    service: GenreClassifierPort | None = getattr(request.app.state, "classifier", None)
    if service is None or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classifier model is not available",
        )
    return service


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Formats unhandled exceptions into standardized ErrorResponse JSON."""
    logger.error(f"Unhandled error: {exc}")
    payload = ErrorResponse(
        error="InternalServerError",
        message="Internal server error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details={"type": type(exc).__name__, "message": str(exc)},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=payload.model_dump(),
    )


@app.get("/", summary="Root service metadata")
async def get_root_info() -> Dict[str, str]:
    """Returns basic API metadata and navigation links."""
    return {
        "message": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "genres": "/genres",
        "classify": "/classify",
    }


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def get_health_status(request: Request) -> HealthResponse:
    """Reports service health, loaded model status, and GPU memory metrics."""
    service: GenreClassifierPort | None = getattr(request.app.state, "classifier", None)
    model_loaded = bool(service and service.is_loaded)
    gpu_memory = await service.get_gpu_memory_info() if model_loaded and service else None

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        version=settings.app_version,
        model_loaded=model_loaded,
        device=settings.device,
        gpu_memory=gpu_memory,
        available_genres=settings.available_genres,
    )


@app.get("/genres", response_model=GenreListResponse, summary="List supported card genres")
async def list_available_genres() -> GenreListResponse:
    """Returns the 11 canonical Yu-Gi-Oh! card mechanics genres."""
    return GenreListResponse(
        genres=settings.available_genres,
        total=len(settings.available_genres),
    )


@app.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify card effect descriptions into genres",
)
async def classify_card_descriptions(
    payload: ClassificationRequest,
    classifier: GenreClassifierPort = Depends(get_classifier_service),
) -> ClassificationResponse:
    """Classifies between 1 and 50 card descriptions into multi-label genres."""
    active_threshold = (
        payload.threshold if payload.threshold is not None else settings.threshold
    )
    start_time = time.time()
    results = await classifier.classify_batch(payload.texts, active_threshold)
    elapsed = time.time() - start_time

    return ClassificationResponse(
        results=results,
        statistics=BatchStatisticsCalculator.calculate_statistics(
            results, elapsed, active_threshold
        ),
        metadata=BatchStatisticsCalculator.build_metadata(),
    )


@app.get("/stats", summary="Get model runtime statistics")
async def get_runtime_stats(
    classifier: GenreClassifierPort = Depends(get_classifier_service),
) -> Dict[str, Any]:
    """Returns cumulative inference statistics and hardware utilization."""
    return await classifier.get_model_stats()
