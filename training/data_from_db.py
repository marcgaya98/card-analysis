"""Extracts card descriptions and genres from MySQL for multi-label training."""
import os
from typing import Optional
import mysql.connector
import pandas as pd
from dotenv import load_dotenv

from app.config import CANONICAL_CARD_GENRES
from app.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

DEFAULT_OUTPUT_PATH = "training/data/cards_db.csv"
ALLOWED_GENRE_SET = set(CANONICAL_CARD_GENRES)

EXTRACTION_QUERY = """
    SELECT c.description, GROUP_CONCAT(g.name ORDER BY g.name SEPARATOR ',') AS genres
    FROM card c
    JOIN card_genres cg ON c.id = cg.cardId
    JOIN genres g ON cg.genreId = g.id
    WHERE c.description IS NOT NULL AND c.description != ''
    GROUP BY c.id
    HAVING genres IS NOT NULL AND genres != ''
"""


def filter_allowed_genres(raw_genres: str) -> str:
    """Keeps only canonical Yu-Gi-Oh! card mechanics genres."""
    valid = [g.strip() for g in raw_genres.split(",") if g.strip() in ALLOWED_GENRE_SET]
    return ",".join(valid)


def open_mysql_connection():
    """Creates a MySQL connection from environment variables."""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME", "yugioh_db"),
        user=os.getenv("DB_USER", "yugioh_user"),
        password=os.getenv("DB_PASSWORD", os.getenv("DB_PASS", "")),
    )


def fetch_card_data(
    save_csv: bool = True, output_path: Optional[str] = None
) -> pd.DataFrame:
    """Extracts and cleans (description, genres) rows from MySQL."""
    conn = open_mysql_connection()
    try:
        dataframe = pd.read_sql(EXTRACTION_QUERY, conn)
    finally:
        conn.close()

    dataframe = dataframe[dataframe["description"].str.strip() != ""]
    dataframe["genres"] = dataframe["genres"].apply(filter_allowed_genres)
    dataframe = dataframe[dataframe["genres"].str.strip() != ""]

    if save_csv:
        destination = output_path or DEFAULT_OUTPUT_PATH
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        dataframe.to_csv(destination, index=False)
        logger.info(f"Saved {len(dataframe)} rows to {destination}")

    return dataframe


if __name__ == "__main__":
    fetch_card_data(save_csv=True)
