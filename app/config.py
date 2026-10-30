"""Application and training configuration with automatic hardware detection."""
import os
from pathlib import Path
from typing import List, Optional
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_CARD_GENRES: List[str] = [
    "Remove card from play",
    "Destroy Spell/Trap",
    "Destroy monster",
    "Attribute",
    "Type",
    "Search Deck",
    "Recover cards from Graveyard",
    "Increase/Decrease ATK/DEF",
    "Reduce Life Points",
    "Special Summon",
    "Cannot perform Normal Summon",
]


def resolve_default_device() -> str:
    """Selects CUDA only when enabled via env and physically available in PyTorch."""
    cuda_env_enabled = os.environ.get("CUDA_AVAILABLE", "true").lower() == "true"
    if cuda_env_enabled and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or .env file."""

    app_name: str = "Yu-Gi-Oh! Card Genre Classifier API"
    app_version: str = "2.0.0"
    debug: bool = False

    model_name: str = Field(default="roberta-base")
    model_path: str = Field(
        default=str(PROJECT_ROOT / "models" / "card-genre-classifier")
    )
    max_length: int = Field(default=512, ge=32, le=1024)

    batch_size: int = Field(default=32, ge=1, le=128)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    device: str = Field(default_factory=resolve_default_device)

    use_mixed_precision: bool = Field(default=True)
    gradient_checkpointing: bool = Field(default=True)
    available_genres: List[str] = Field(
        default_factory=lambda: list(CANONICAL_CARD_GENRES)
    )

    log_level: str = "INFO"
    database_url: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()


class TrainingConfig:
    """Hyperparameters and directory settings for model training."""

    MODEL_NAME: str = settings.model_name
    MAX_LENGTH: int = settings.max_length
    NUM_LABELS: int = len(CANONICAL_CARD_GENRES)

    LEARNING_RATE: float = 2e-5
    BATCH_SIZE: int = settings.batch_size
    GRADIENT_ACCUMULATION_STEPS: int = 2
    NUM_EPOCHS: int = 3
    WARMUP_STEPS: int = 500
    WEIGHT_DECAY: float = 0.01

    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.1
    TEST_SPLIT: float = 0.1

    MODEL_OUTPUT_DIR: str = "./models/card-genre-classifier"
    LOGS_DIR: str = "./logs"

    USE_GRADIENT_CHECKPOINTING: bool = True
    USE_MIXED_PRECISION: bool = False
    DATALOADER_NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True


training_config = TrainingConfig()
