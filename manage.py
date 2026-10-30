#!/usr/bin/env python3
"""CLI management utility for the Yu-Gi-Oh! Card Genre Classifier."""
import json
import subprocess
import sys
from pathlib import Path
import click
import torch
import uvicorn

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


@click.group()
def cli() -> None:
    """Yu-Gi-Oh! Card Genre Classifier CLI."""


@cli.command()
@click.option("--output", "-o", default="training/data/cards_template.csv")
def create_template(output: str) -> None:
    """Create a sample CSV template for training data."""
    from training.data_preparation import create_training_data_template

    template_path = create_training_data_template(output)
    click.echo(f"Template created at: {template_path}")


@cli.command()
@click.argument("data_path", default="training/data/cards_db.csv")
def validate_data(data_path: str) -> None:
    """Validate a training CSV/JSON dataset."""
    from training.data_preparation import validate_training_data

    results = validate_training_data(data_path)
    click.echo(json.dumps(results, indent=2))


@cli.command()
@click.option("--data-path", "-d", default="training/data/cards_db.csv")
@click.option("--epochs", "-e", default=3)
@click.option("--batch-size", "-b", default=16)
@click.option("--output-dir", "-o", default=None)
def train(data_path: str, epochs: int, batch_size: int, output_dir: str | None) -> None:
    """Train the multi-label genre classification model."""
    from app.config import training_config
    from training.train import CardGenreTrainer

    training_config.NUM_EPOCHS = epochs
    training_config.BATCH_SIZE = batch_size

    trainer = CardGenreTrainer(data_path, output_dir)
    texts, genres_lists = trainer.load_and_prepare_data()
    trainer.prepare_tokenizer()
    trainer.create_datasets(texts, genres_lists)
    trainer.prepare_model()
    trainer.train()
    click.echo(f"Training completed. Model saved at: {trainer.output_dir}")


@cli.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8000)
@click.option("--reload", is_flag=True)
def serve(host: str, port: int, reload: bool) -> None:
    """Start the FastAPI server."""
    uvicorn.run("app.main:app", host=host, port=port, reload=reload, log_level="info")


@cli.command()
def test() -> None:
    """Run the pytest test suite."""
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)


@cli.command()
def check_gpu() -> None:
    """Check PyTorch CUDA availability and GPU device details."""
    click.echo(f"PyTorch version: {torch.__version__}")
    click.echo(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"Device: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    cli()
