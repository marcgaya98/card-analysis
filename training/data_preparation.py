"""Training dataset template generation and validation utilities."""
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from app.config import CANONICAL_CARD_GENRES
from app.utils.logger import get_logger

logger = get_logger(__name__)

SAMPLE_CARDS_BY_GENRE: List[Dict[str, str]] = [
    {
        "description": "Banish 1 monster on the field.",
        "genres": "Remove card from play",
    },
    {
        "description": "Target 1 Spell/Trap on the field; destroy that target.",
        "genres": "Destroy Spell/Trap",
    },
    {
        "description": "Destroy all monsters your opponent controls.",
        "genres": "Destroy monster",
    },
    {
        "description": "All LIGHT monsters on the field gain 500 ATK and DARK monsters lose 400 ATK.",
        "genres": "Attribute,Increase/Decrease ATK/DEF",
    },
    {
        "description": "All Warrior-Type monsters you control gain 400 ATK.",
        "genres": "Type,Increase/Decrease ATK/DEF",
    },
    {
        "description": "Add 1 Level 4 or lower monster from your Deck to your hand.",
        "genres": "Search Deck",
    },
    {
        "description": "Target 1 monster in your Graveyard; add it to your hand.",
        "genres": "Recover cards from Graveyard",
    },
    {
        "description": "Inflict 1000 damage to your opponent's Life Points.",
        "genres": "Reduce Life Points",
    },
    {
        "description": "Cannot be Normal Summoned/Set. Must be Special Summoned from your hand.",
        "genres": "Cannot perform Normal Summon,Special Summon",
    },
]


def create_training_data_template(
    output_path: str = "training/data/cards_template.csv",
) -> Path:
    """Creates a CSV template using canonical Yu-Gi-Oh! genres and columns."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(SAMPLE_CARDS_BY_GENRE).to_csv(destination, index=False)
    logger.info(f"Template created at {destination}")
    return destination


def parse_row_genres(raw_genres: Any) -> List[str]:
    """Splits a comma/semicolon-separated genre value into clean strings."""
    if isinstance(raw_genres, list):
        return [str(item).strip() for item in raw_genres if str(item).strip()]
    return [
        item.strip()
        for item in str(raw_genres).replace(";", ",").split(",")
        if item.strip()
    ]


def validate_training_data(data_path: str) -> Dict[str, Any]:
    """Validates a training CSV/JSON file against canonical Yu-Gi-Oh! genres."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = (
        pd.read_csv(path)
        if path.suffix.lower() == ".csv"
        else pd.read_json(path)
    )
    valid_genre_set = set(CANONICAL_CARD_GENRES)
    distribution: Dict[str, int] = {}
    invalid_genres: List[str] = []
    lengths: List[int] = []
    valid_rows = 0
    missing_desc = 0
    missing_genres = 0

    for _, row in df.iterrows():
        desc = row.get("description")
        raw_genres = row.get("genres")
        if pd.isna(desc) or not str(desc).strip():
            missing_desc += 1
            continue
        if pd.isna(raw_genres) or not str(raw_genres).strip():
            missing_genres += 1
            continue

        lengths.append(len(str(desc)))
        row_genres = parse_row_genres(raw_genres)
        unknown = [g for g in row_genres if g not in valid_genre_set]
        if unknown:
            invalid_genres.extend(unknown)
            continue

        valid_rows += 1
        for genre in row_genres:
            distribution[genre] = distribution.get(genre, 0) + 1

    return {
        "total_samples": len(df),
        "valid_samples": valid_rows,
        "invalid_samples": len(df) - valid_rows,
        "missing_descriptions": missing_desc,
        "missing_genres": missing_genres,
        "unique_invalid_genres": sorted(set(invalid_genres)),
        "genre_distribution": distribution,
        "avg_text_length": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
    }


if __name__ == "__main__":
    report = validate_training_data("training/data/cards_db.csv")
    print(json.dumps(report, indent=2))
