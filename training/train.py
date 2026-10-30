"""Multi-label Yu-Gi-Oh! card genre classification training pipeline."""
import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from app.config import CANONICAL_CARD_GENRES, training_config
from app.utils.logger import get_logger
from training.data_preparation import parse_row_genres
from training.weighted_loss import WeightedBCEWithLogitsLoss, get_pos_weights_tensor

logger = get_logger(__name__)
ALLOWED_GENRES_SET = set(CANONICAL_CARD_GENRES)


class CardGenreDataset(Dataset):
    """PyTorch Dataset wrapping card descriptions and multi-hot genre vectors."""

    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenizer,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class WeightedTrainer(Trainer):
    """Hugging Face Trainer with custom class-weighted BCE loss."""

    def __init__(self, *args, loss_fn: Optional[nn.Module] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = (
            self.loss_fn(logits, labels, reduction="mean")
            if self.loss_fn is not None
            else nn.functional.binary_cross_entropy_with_logits(
                logits, labels, reduction="mean"
            )
        )
        return (loss, outputs) if return_outputs else loss


class CardGenreTrainer:
    """Orchestrates dataset preparation, model initialization, and fine-tuning."""

    def __init__(self, data_path: str, output_dir: Optional[str] = None) -> None:
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir or training_config.MODEL_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = None
        self.model = None
        self.mlb = MultiLabelBinarizer(classes=CANONICAL_CARD_GENRES)
        self.genres_lists: List[List[str]] = []
        self.train_dataset: Optional[CardGenreDataset] = None
        self.val_dataset: Optional[CardGenreDataset] = None
        self.test_dataset: Optional[CardGenreDataset] = None

    def load_and_prepare_data(self) -> Tuple[List[str], List[List[str]]]:
        """Loads and atomically filters aligned (description, valid_genres) pairs."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Training dataset not found: {self.data_path}")

        df = (
            pd.read_csv(self.data_path)
            if self.data_path.suffix.lower() == ".csv"
            else pd.read_json(self.data_path)
        )
        texts, genres_lists = extract_valid_samples(df, CANONICAL_CARD_GENRES)
        if not texts:
            raise ValueError("No valid training rows found after genre filtering.")

        self.genres_lists = genres_lists
        logger.info(f"Loaded {len(texts)} aligned training samples")
        return texts, genres_lists


def extract_valid_samples(
    df: pd.DataFrame, allowed_genres: Optional[List[str]] = None
) -> Tuple[List[str], List[List[str]]]:
    allowed_set = set(allowed_genres) if allowed_genres is not None else ALLOWED_GENRES_SET
    texts: List[str] = []
    genres_lists: List[List[str]] = []

    for _, row in df.iterrows():
        description = str(row.get("description", "")).strip()
        valid_genres = [
            g for g in parse_row_genres(row.get("genres", "")) if g in allowed_set
        ]
        if not description or not valid_genres:
            continue
        texts.append(description)
        genres_lists.append(valid_genres)

    return texts, genres_lists

    def prepare_tokenizer(self) -> None:
        """Loads the Hugging Face tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(training_config.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_model(self) -> None:
        """Initializes the sequence classification model for multi-label training."""
        num_labels = len(CANONICAL_CARD_GENRES)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            training_config.MODEL_NAME,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True,
        )
        if training_config.USE_GRADIENT_CHECKPOINTING:
            self.model.gradient_checkpointing_enable()

    def create_datasets(
        self, texts: List[str], genres_lists: List[List[str]]
    ) -> None:
        """Splits data into train/val/test datasets and writes genre_mapping.json."""
        encoded_labels = self.mlb.fit_transform(genres_lists)
        holdout_ratio = training_config.VAL_SPLIT + training_config.TEST_SPLIT

        x_train, x_temp, y_train, y_temp = train_test_split(
            texts, encoded_labels, test_size=holdout_ratio, random_state=42
        )
        val_ratio = training_config.VAL_SPLIT / holdout_ratio
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=(1 - val_ratio), random_state=42
        )

        max_len = training_config.MAX_LENGTH
        self.train_dataset = CardGenreDataset(x_train, y_train.tolist(), self.tokenizer, max_len)
        self.val_dataset = CardGenreDataset(x_val, y_val.tolist(), self.tokenizer, max_len)
        self.test_dataset = CardGenreDataset(x_test, y_test.tolist(), self.tokenizer, max_len)

        with open(self.output_dir / "genre_mapping.json", "w", encoding="utf-8") as file:
            json.dump(
                {"classes": self.mlb.classes_.tolist(), "num_classes": len(self.mlb.classes_)},
                file,
                indent=2,
            )

    def train(self) -> Trainer:
        """Executes fine-tuning and saves model artifacts and summary metrics."""
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=training_config.NUM_EPOCHS,
            per_device_train_batch_size=training_config.BATCH_SIZE,
            per_device_eval_batch_size=training_config.BATCH_SIZE,
            gradient_accumulation_steps=training_config.GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=training_config.WARMUP_STEPS,
            weight_decay=training_config.WEIGHT_DECAY,
            learning_rate=training_config.LEARNING_RATE,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=training_config.USE_MIXED_PRECISION,
            report_to=None,
            seed=42,
            remove_unused_columns=False,
        )
        pos_weights = get_pos_weights_tensor(self.genres_lists, self.mlb, self.model.device)
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            loss_fn=WeightedBCEWithLogitsLoss(pos_weights),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        start_time = time.time()
        trainer.train()
        duration_minutes = (time.time() - start_time) / 60.0

        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        eval_metrics = trainer.evaluate(self.test_dataset)
        self._save_training_summary(duration_minutes, eval_metrics.get("eval_loss"))
        return trainer

    def _save_training_summary(
        self, duration_minutes: float, final_eval_loss: Optional[float]
    ) -> None:
        summary = {
            "model_name": training_config.MODEL_NAME,
            "num_labels": len(CANONICAL_CARD_GENRES),
            "genres": CANONICAL_CARD_GENRES,
            "training_time_minutes": duration_minutes,
            "final_eval_loss": final_eval_loss,
            "train_samples": len(self.train_dataset) if self.train_dataset else 0,
            "val_samples": len(self.val_dataset) if self.val_dataset else 0,
            "test_samples": len(self.test_dataset) if self.test_dataset else 0,
        }
        with open(self.output_dir / "training_summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Yu-Gi-Oh! card genre classifier")
    parser.add_argument("--data-path", type=str, default="training/data/cards_db.csv")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    trainer = CardGenreTrainer(args.data_path, args.output_dir)
    texts, genres_lists = trainer.load_and_prepare_data()
    trainer.prepare_tokenizer()
    trainer.create_datasets(texts, genres_lists)
    trainer.prepare_model()
    trainer.train()


if __name__ == "__main__":
    main()
