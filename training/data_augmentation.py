import random
from collections import Counter
from typing import Dict, List

import pandas as pd

YUGIOH_TERM_SYNONYMS: Dict[str, List[str]] = {
    "destroy": ["eliminate", "remove", "banish"],
    "monster": ["creature", "card"],
    "opponent": ["enemy", "foe"],
    "damage": ["harm", "injury"],
    "summon": ["call", "invoke"],
    "graveyard": ["GY", "discard pile"],
    "field": ["board", "area"],
    "hand": ["cards in hand"],
    "deck": ["main deck"],
    "turn": ["phase"],
}

SYNONYM_REPLACEMENT_PROBABILITY = 0.3
VARIATIONS_PER_DESCRIPTION = 2
DEFAULT_TARGET_MIN_COUNT = 95


def generate_single_paraphrase(description: str) -> str:
    variation = description.lower()
    for original, replacements in YUGIOH_TERM_SYNONYMS.items():
        if original in variation and random.random() < SYNONYM_REPLACEMENT_PROBABILITY:
            replacement = random.choice(replacements)
            variation = variation.replace(original, replacement, 1)

    if not variation:
        return variation
    return variation[0].upper() + variation[1:]


def paraphrase_description(description: str) -> List[str]:
    variations: List[str] = []
    for _ in range(VARIATIONS_PER_DESCRIPTION):
        variation = generate_single_paraphrase(description)
        if variation != description and variation not in variations:
            variations.append(variation)
    return variations


def count_genres_in_dataframe(df: pd.DataFrame) -> Counter:
    all_genres: List[str] = []
    for genres in df["genres"].dropna():
        all_genres.extend(genre.strip() for genre in str(genres).split(",") if genre.strip())
    return Counter(all_genres)


def generate_augmented_rows_for_genre(
    genre_cards: pd.DataFrame, needed_count: int
) -> List[Dict[str, str]]:
    augmented_rows: List[Dict[str, str]] = []
    for _, row in genre_cards.iterrows():
        if len(augmented_rows) >= needed_count:
            break
        for variation in paraphrase_description(str(row["description"])):
            if len(augmented_rows) >= needed_count:
                break
            augmented_rows.append({"description": variation, "genres": str(row["genres"])})
    return augmented_rows


def augment_minority_classes(
    df: pd.DataFrame, target_min_count: int = DEFAULT_TARGET_MIN_COUNT
) -> pd.DataFrame:
    genre_counts = count_genres_in_dataframe(df)
    augmented_rows: List[Dict[str, str]] = []

    for genre, count in genre_counts.items():
        if count >= target_min_count:
            continue
        needed = target_min_count - count
        genre_cards = df[df["genres"].str.contains(genre, regex=False, na=False)]
        augmented_rows.extend(generate_augmented_rows_for_genre(genre_cards, needed))

    if not augmented_rows:
        return df

    augmented_df = pd.DataFrame(augmented_rows)
    return pd.concat([df, augmented_df], ignore_index=True)


def balance_dataset(input_file: str, output_file: str, method: str = "weighted") -> pd.DataFrame:
    df = pd.read_csv(input_file)

    if method == "augmentation":
        balanced_df = augment_minority_classes(df)
        balanced_df.to_csv(output_file, index=False)
        return balanced_df

    return df
