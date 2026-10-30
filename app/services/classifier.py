"""Multi-label card genre classification service and domain contracts."""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from app.config import settings
from app.schemas import GenrePrediction, SingleClassificationResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

HIGH_CONFIDENCE_THRESHOLD = 0.8
MEDIUM_CONFIDENCE_THRESHOLD = 0.6
BYTES_PER_GIGABYTE = 1024**3
DEFAULT_SEED = 42


class GenreClassifierPort(Protocol):
    """Port interface for card genre classification."""

    is_loaded: bool

    async def load_model(self) -> None: ...
    async def classify_batch(
        self, texts: List[str], threshold: Optional[float] = None
    ) -> List[SingleClassificationResult]: ...
    async def get_gpu_memory_info(self) -> Optional[Dict[str, str]]: ...
    async def get_model_stats(self) -> Dict[str, Any]: ...
    async def cleanup(self) -> None: ...


class BatchStatisticsCalculator:
    """Pure calculator for batch classification metrics and metadata."""

    @staticmethod
    def calculate_statistics(
        results: List[SingleClassificationResult],
        total_time: float,
        threshold: float,
    ) -> Dict[str, Any]:
        count = len(results)
        if count == 0:
            return {"processed_texts": 0, "total_time_seconds": 0.0}

        total_primary = sum(len(item.primary_genres) for item in results)
        avg_confidence = (
            sum(
                sum(pred.probability for pred in item.detected_genres)
                / max(len(item.detected_genres), 1)
                for item in results
            )
            / count
        )
        return {
            "processed_texts": count,
            "total_time_seconds": round(total_time, 3),
            "average_time_seconds": round(total_time / count, 3),
            "total_primary_genres_detected": total_primary,
            "average_confidence": round(avg_confidence, 3),
            "threshold_used": threshold,
        }

    @staticmethod
    def build_metadata() -> Dict[str, Any]:
        return {
            "model": settings.model_name,
            "device": settings.device,
            "batch_size": settings.batch_size,
            "max_length": settings.max_length,
            "api_version": settings.app_version,
        }


def resolve_confidence_label(probability: float) -> str:
    """Maps a probability score to a human-readable confidence tier."""
    if probability >= HIGH_CONFIDENCE_THRESHOLD:
        return "High"
    if probability >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "Medium"
    return "Low"


class CardGenreClassifier:
    """PyTorch/Transformers adapter for multi-label Yu-Gi-Oh! genre classification."""

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.device: str = settings.device
        self.is_loaded: bool = False
        self.genre_labels: List[str] = list(settings.available_genres)
        self.classification_count: int = 0
        self.total_inference_time: float = 0.0
        set_seed(DEFAULT_SEED)

    async def load_model(self) -> None:
        """Loads either the fine-tuned model from disk or falls back to the base model."""
        model_path = Path(settings.model_path)
        self.genre_labels = self._load_genre_mapping(model_path)

        if (model_path / "config.json").exists() and (
            model_path / "model.safetensors"
        ).exists():
            self._load_pretrained_checkpoint(str(model_path), ignore_mismatched=False)
        else:
            self._load_pretrained_checkpoint(settings.model_name, ignore_mismatched=True)

        self.is_loaded = True
        logger.info(f"Model loaded on device={self.device} ({len(self.genre_labels)} genres)")

    def _load_genre_mapping(self, model_path: Path) -> List[str]:
        mapping_file = model_path / "genre_mapping.json"
        if not mapping_file.exists():
            return list(settings.available_genres)
        with open(mapping_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get("classes", list(settings.available_genres))

    def _load_pretrained_checkpoint(self, source: str, ignore_mismatched: bool) -> None:
        use_fp16 = settings.use_mixed_precision and self.device == "cuda"
        dtype = torch.float16 if use_fp16 else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            source,
            num_labels=len(self.genre_labels),
            problem_type="multi_label_classification",
            torch_dtype=dtype,
            ignore_mismatched_sizes=ignore_mismatched,
        ).to(self.device)
        self.model.eval()

    async def classify_batch(
        self, texts: List[str], threshold: Optional[float] = None
    ) -> List[SingleClassificationResult]:
        """Classifies a list of card descriptions in memory-safe mini-batches."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Classifier model is not loaded")

        active_threshold = threshold if threshold is not None else settings.threshold
        results: List[SingleClassificationResult] = []
        step = settings.batch_size

        for start_idx in range(0, len(texts), step):
            chunk = texts[start_idx : start_idx + step]
            results.extend(self._infer_chunk(chunk, active_threshold))

        self.classification_count += len(texts)
        return results

    def _infer_chunk(
        self, texts: List[str], threshold: float
    ) -> List[SingleClassificationResult]:
        start_time = time.time()
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=settings.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits).cpu().numpy()

        elapsed = time.time() - start_time
        self.total_inference_time += elapsed
        per_item_time = elapsed / max(len(texts), 1)

        return [
            self._build_single_result(text, probs, threshold, per_item_time)
            for text, probs in zip(texts, probabilities)
        ]

    def _build_single_result(
        self, text: str, probabilities: Any, threshold: float, item_time: float
    ) -> SingleClassificationResult:
        predictions: List[GenrePrediction] = []
        primary_genres: List[str] = []

        for genre, raw_prob in zip(self.genre_labels, probabilities):
            prob = float(raw_prob)
            predictions.append(
                GenrePrediction(
                    genre=genre,
                    probability=prob,
                    confidence=resolve_confidence_label(prob),
                )
            )
            if prob >= threshold:
                primary_genres.append(genre)

        predictions.sort(key=lambda item: item.probability, reverse=True)
        return SingleClassificationResult(
            text=text,
            detected_genres=predictions,
            primary_genres=primary_genres,
            processing_time=round(item_time, 4),
        )

    async def get_gpu_memory_info(self) -> Optional[Dict[str, str]]:
        """Returns formatted CUDA memory metrics when running on GPU."""
        if not torch.cuda.is_available():
            return None

        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        total = torch.cuda.get_device_properties(0).total_memory
        return {
            "allocated": f"{allocated / BYTES_PER_GIGABYTE:.2f} GB",
            "cached": f"{reserved / BYTES_PER_GIGABYTE:.2f} GB",
            "total": f"{total / BYTES_PER_GIGABYTE:.2f} GB",
            "free": f"{(total - allocated) / BYTES_PER_GIGABYTE:.2f} GB",
            "utilization": f"{(allocated / total) * 100:.1f}%",
        }

    async def get_model_stats(self) -> Dict[str, Any]:
        """Returns cumulative runtime statistics and model configuration."""
        avg_time = (
            self.total_inference_time / self.classification_count
            if self.classification_count > 0
            else 0.0
        )
        stats: Dict[str, Any] = {
            "model_loaded": self.is_loaded,
            "total_classifications": self.classification_count,
            "total_inference_time_seconds": round(self.total_inference_time, 3),
            "average_time_per_text_seconds": round(avg_time, 4),
            "device": self.device,
            "available_genres_count": len(self.genre_labels),
            "configuration": {
                "model": settings.model_name,
                "max_length": settings.max_length,
                "batch_size": settings.batch_size,
                "threshold": settings.threshold,
                "mixed_precision": settings.use_mixed_precision,
            },
        }
        if self.device == "cuda":
            stats["gpu_memory"] = await self.get_gpu_memory_info()
        return stats

    async def cleanup(self) -> None:
        """Releases model tensors and clears CUDA cache."""
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
