up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

format:
	uv run ruff check --select I --fix
	uv run ruff format

transform:
	uv run scraper/booking/transform.py