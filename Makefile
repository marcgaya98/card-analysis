PROFILE ?= cpu

.PHONY: build up down restart logs test lint train validate

build:
	docker compose --profile $(PROFILE) build

up:
	docker compose --profile $(PROFILE) up -d

down:
	docker compose --profile $(PROFILE) down

restart:
	docker compose --profile $(PROFILE) restart

logs:
	docker compose --profile $(PROFILE) logs -f

test:
	pytest tests/ -v

lint:
	black --check app training tests && flake8 app training tests

validate:
	python manage.py validate-data --data-file training/data/cards_db.csv

train:
	python manage.py train --data-path training/data/cards_db.csv
