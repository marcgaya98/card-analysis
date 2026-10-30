# 🃏 Yu-Gi-Oh! Card Genre Classifier API

Multi-label deep learning classifier for **Yu-Gi-Oh! card effect mechanics** built with **PyTorch**, **Hugging Face Transformers (`roberta-base`)**, and **FastAPI**.

Given one or more card effect descriptions, the model predicts probabilities across **11 canonical Yu-Gi-Oh! card mechanics genres** with automatic GPU (`CUDA` + FP16 mixed precision) or CPU fallback.

---

## 🎯 Supported Card Genres (11 Mechanics)

| Genre | Description |
|---|---|
| `Remove card from play` | Effects that banish cards from the field, hand, Deck, or Graveyard |
| `Destroy Spell/Trap` | Destruction of Spell and Trap cards |
| `Destroy monster` | Destruction of monster cards by effect |
| `Attribute` | Effects targeting or modifying specific Monster Attributes (`LIGHT`, `DARK`, `FIRE`, etc.) |
| `Type` | Effects targeting or supporting specific Monster Types (`Warrior`, `Spellcaster`, `Pyro`, etc.) |
| `Search Deck` | Adding or excavating cards from the Main Deck |
| `Recover cards from Graveyard` | Returning or retrieving cards from the Graveyard (`GY`) |
| `Increase/Decrease ATK/DEF` | Stat modification (`ATK` / `DEF` boosts or reductions) |
| `Reduce Life Points` | Direct burn damage or Life Point reduction effects |
| `Special Summon` | Special Summoning monsters from hand, Deck, Extra Deck, or Graveyard |
| `Cannot perform Normal Summon` | Summoning restrictions (`Cannot be Normal Summoned/Set`, `Must first be Special Summoned`) |

---

## 🏗️ Architecture & Project Structure

```text
card-analysis/
├── app/                          # FastAPI Application Layer
│   ├── config.py                 # Single source of truth for genres, Settings, and TrainingConfig
│   ├── main.py                   # REST API endpoints with FastAPI Dependency Injection
│   ├── schemas/                  # Standardized English Pydantic v2 DTOs
│   ├── services/
│   │   └── classifier.py         # GenreClassifierPort, CardGenreClassifier, BatchStatisticsCalculator
│   └── utils/
│       └── logger.py             # Structured Loguru logger
├── training/                     # Model Training & Dataset Pipeline
│   ├── train.py                  # RoBERTa multi-label fine-tuning with Weighted BCE Loss
│   ├── weighted_loss.py          # Class-balanced BCEWithLogitsLoss for imbalanced genres
│   ├── data_from_db.py           # MySQL/MariaDB dataset extractor
│   ├── data_preparation.py       # Dataset validator and CSV template generator
│   ├── data_augmentation.py      # Minority-class synonym paraphrasing utility
│   └── data/
│       └── cards_db.csv          # Curated Yu-Gi-Oh! card descriptions and multi-label genres
├── models/
│   └── card-genre-classifier/    # Fine-tuned roberta-base model & genre_mapping.json
├── tests/
│   └── test_api.py               # Automated API, schema validation, and dataset alignment tests
├── Dockerfile                    # Multi-stage Python 3.11 container image
├── docker-compose.yml            # Clean CPU & GPU container profiles
├── manage.py                     # CLI management entrypoint (serve, train, validate-data, test-api)
└── requirements.txt              # Python dependencies
```

---

## 🚀 Quick Start (Local Development)

### 1. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Start the FastAPI Server

```bash
# Using manage.py CLI
python manage.py serve --reload

# Or directly with Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 📡 REST API Endpoints

### `POST /classify` — Multi-Label Card Classification

Classifies between `1` and `50` card descriptions in a single batch.

```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Once per turn, during your next Standby Phase after this card was destroyed by a card effect and sent to the GY: Special Summon this card from the GY. If you do: Destroy all Spells and Traps on the field.",
      "All FIRE monsters on the field gain 500 ATK. All WATER monsters on the field lose 400 ATK."
    ],
    "threshold": 0.5
  }'
```

**Response (`200 OK`)**:
```json
{
  "results": [
    {
      "text": "Once per turn, during your next Standby Phase...",
      "detected_genres": [
        {
          "genre": "Special Summon",
          "probability": 0.9124,
          "confidence": "High"
        },
        {
          "genre": "Destroy Spell/Trap",
          "probability": 0.8741,
          "confidence": "High"
        }
      ],
      "primary_genres": ["Special Summon", "Destroy Spell/Trap"],
      "processing_time": 0.0182
    }
  ],
  "statistics": {
    "total_texts": 2,
    "texts_with_genres": 2,
    "texts_without_genres": 0,
    "average_genres_per_text": 2.0,
    "most_common_genres": [["Special Summon", 1], ["Destroy Spell/Trap", 1]],
    "average_confidence": 0.893
  },
  "metadata": {
    "model_used": "roberta-base",
    "threshold": 0.5,
    "total_processing_time_seconds": 0.0364,
    "device": "cuda",
    "available_genres_count": 11
  }
}
```

### Other Endpoints
- `GET /` — API overview and endpoint map
- `GET /health` — Readiness, device (`cuda` / `cpu`), GPU VRAM usage, and loaded genres
- `GET /genres` — List of all 11 supported Yu-Gi-Oh! card genres
- `GET /stats` — Model configuration and runtime metrics

---

## 🎓 Model Training Pipeline

```bash
# 1. Validate training dataset against canonical genres
python manage.py validate-data --data-file training/data/cards_db.csv

# 2. Optional: Extract fresh card data from MySQL/MariaDB (uses DB_* in .env)
python -m training.data_from_db

# 3. Train multi-label RoBERTa model with class-weighted BCE loss
python manage.py train --data-path training/data/cards_db.csv --epochs 12 --batch-size 16
```

---

## 🐳 Docker Usage

The root `docker-compose.yml` provides clean `cpu` (default) and `gpu` (NVIDIA Runtime) profiles:

```bash
# Run on CPU (default profile, port 8000)
docker compose up -d --build

# Run with NVIDIA GPU acceleration
docker compose --profile gpu up -d --build
```

---

## 🧪 Automated Testing

```bash
pytest tests/ -v
```
