import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import CANONICAL_CARD_GENRES, PROJECT_ROOT, settings
from app.main import app
from app.schemas import ClassificationRequest
from training.data_preparation import validate_training_data
from training.train import extract_valid_samples


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as client:
        yield client


class TestClassificationAPI:
    def test_root_endpoint(self, api_client: TestClient):
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["classify"] == "/classify"
        assert data["genres"] == "/genres"
        assert data["health"] == "/health"

    def test_health_endpoint(self, api_client: TestClient):
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "version" in data
        assert "model_loaded" in data
        assert "device" in data
        assert data["available_genres"] == CANONICAL_CARD_GENRES

    def test_genres_endpoint(self, api_client: TestClient):
        response = api_client.get("/genres")
        assert response.status_code == 200
        data = response.json()
        assert "genres" in data
        assert "total" in data
        assert isinstance(data["genres"], list)
        assert data["total"] == len(CANONICAL_CARD_GENRES)
        assert set(data["genres"]) == set(CANONICAL_CARD_GENRES)

    def test_stats_endpoint(self, api_client: TestClient):
        response = api_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "device" in data
        assert "configuration" in data
        assert data["available_genres_count"] == len(CANONICAL_CARD_GENRES)

    def test_classify_single_and_batch_descriptions(self, api_client: TestClient):
        descriptions = [
            "Once per turn, during your next Standby Phase after this card was destroyed by a card effect and sent to the GY: Special Summon this card from the GY. If you do: Destroy all Spells and Traps on the field.",
            "Cannot be Normal Summoned/Set. Must first be Special Summoned (from your hand) by banishing 1 FIRE monster from your GY. If this card destroys an opponent's monster by battle and sends it to the GY: Inflict 1500 damage to your opponent.",
            "All FIRE monsters on the field gain 500 ATK. All WATER monsters on the field lose 400 ATK.",
        ]
        response = api_client.post(
            "/classify",
            json={"texts": descriptions, "threshold": 0.4},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "statistics" in data
        assert "metadata" in data
        assert len(data["results"]) == len(descriptions)

        first_result = data["results"][0]
        assert "text" in first_result
        assert "detected_genres" in first_result
        assert "primary_genres" in first_result
        assert "processing_time" in first_result
        assert len(first_result["primary_genres"]) > 0

    def test_classify_invalid_input(self, api_client: TestClient):
        empty_list_response = api_client.post("/classify", json={"texts": []})
        assert empty_list_response.status_code == 422

        missing_field_response = api_client.post("/classify", json={})
        assert missing_field_response.status_code == 422

        invalid_threshold_response = api_client.post(
            "/classify",
            json={"texts": ["Destroy 1 monster on the field."], "threshold": 1.5},
        )
        assert invalid_threshold_response.status_code == 422


class TestSchemaValidation:
    def test_valid_request_schema(self):
        request = ClassificationRequest(
            texts=["Destroy all monsters on the field."],
            threshold=0.5,
        )
        assert len(request.texts) == 1
        assert request.threshold == 0.5

    def test_whitespace_only_texts_rejected(self):
        with pytest.raises(ValidationError):
            ClassificationRequest(texts=["   ", "\n"])

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValidationError):
            ClassificationRequest(
                texts=["Destroy all monsters on the field."],
                threshold=2.0,
            )

    def test_exceeding_max_batch_size_rejected(self):
        with pytest.raises(ValidationError):
            ClassificationRequest(texts=["card effect"] * 51)


class TestConfigurationAndTrainingData:
    def test_settings_loaded_with_canonical_genres(self):
        assert settings.app_name is not None
        assert settings.app_version is not None
        assert settings.available_genres == CANONICAL_CARD_GENRES
        assert settings.max_length > 0
        assert settings.batch_size > 0

    def test_training_dataset_validation(self):
        dataset_path = str(PROJECT_ROOT / "training" / "data" / "cards_db.csv")
        report = validate_training_data(dataset_path)
        assert report["valid_samples"] == 854
        assert report["invalid_samples"] == 0
        assert len(report["unique_invalid_genres"]) == 0

    def test_extract_valid_samples_preserves_index_alignment(self):
        sample_df = pd.DataFrame(
            [
                {
                    "description": "Destroy 1 Spell/Trap on the field.",
                    "genres": "Destroy Spell/Trap",
                },
                {
                    "description": "Row with unknown genre that must be skipped.",
                    "genres": "NonExistentGenre",
                },
                {
                    "description": "Special Summon 1 monster from your GY.",
                    "genres": "Special Summon,Recover cards from Graveyard",
                },
            ]
        )
        texts, genre_lists = extract_valid_samples(sample_df, CANONICAL_CARD_GENRES)
        assert len(texts) == len(genre_lists) == 2
        assert texts[0] == "Destroy 1 Spell/Trap on the field."
        assert genre_lists[0] == ["Destroy Spell/Trap"]
        assert texts[1] == "Special Summon 1 monster from your GY."
        assert genre_lists[1] == ["Special Summon", "Recover cards from Graveyard"]
