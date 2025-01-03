up:
	docker compose up -d

down:
	docker compose down

downv:
	docker compose down --volumes

build:
	docker compose build

format:
	uv run ruff check --select I --fix
	uv run ruff format

transform:
	uv run scraper/booking/transform.py

load:
	uv run core/graphdb.py --load

test:
	uv run core/graphdb.py
	
full-load:
	make up
	make load

reload:
	make downv
	make full-load

.PHONY: recommend
recommend:
	uv run core/pipeline.py -m "$(message)"
