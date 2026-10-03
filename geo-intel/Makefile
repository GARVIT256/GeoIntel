# ============================================================
#  GEO-INTEL Makefile
#  Usage: make <target>
#  Requires: Python 3.11, Docker, GNU Make
# ============================================================

PYTHON     := python
PIP        := pip
PYTEST     := pytest
SRC        := src/geointel
CONFIG     := config/config.yaml

.PHONY: help install env data classify change test app db clean

help:
	@echo "GEO-INTEL targets:"
	@echo "  install   - install Python dependencies"
	@echo "  env       - copy .env.example to .env (first-time setup)"
	@echo "  db        - start PostGIS, apply schema, load 1km grid"
	@echo "  data      - fetch/composite Sentinel-2 (uses cache if available)"
	@echo "  classify  - train RF classifier for T1 and T2"
	@echo "  change    - run change detection and association analysis"
	@echo "  test      - run pytest suite"
	@echo "  app       - launch Streamlit demo (offline from cache)"
	@echo "  clean     - remove __pycache__ and .pytest_cache"

install:
	$(PIP) install -e ".[dev]"

env:
	@if not exist .env (copy .env.example .env && echo ".env created – fill in API keys") else (echo ".env already exists")

db:
	docker compose up -d postgis
	@echo "Waiting for PostGIS to be healthy..."
	@$(PYTHON) -c "import time; time.sleep(8)"
	$(PYTHON) -m geointel.db.load --config $(CONFIG)

data:
	$(PYTHON) -m geointel.data.stac_fetch --config $(CONFIG)
	$(PYTHON) -m geointel.data.cloud_mask --config $(CONFIG)
	$(PYTHON) -m geointel.data.composite  --config $(CONFIG)
	$(PYTHON) -m geointel.data.boundary   --config $(CONFIG)
	$(PYTHON) -m geointel.data.osm        --config $(CONFIG)
	$(PYTHON) -m geointel.data.dem        --config $(CONFIG)

classify:
	$(PYTHON) -m geointel.rs.classify_rf --config $(CONFIG)

change:
	$(PYTHON) -m geointel.change.transition   --config $(CONFIG)
	$(PYTHON) -m geointel.change.ndvi_diff    --config $(CONFIG)
	$(PYTHON) -m geointel.change.association  --config $(CONFIG)

test:
	$(PYTEST) tests/ -v --tb=short

app:
	streamlit run $(SRC)/app/streamlit_app.py -- --config $(CONFIG) --offline

clean:
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@if exist .pytest_cache rd /s /q .pytest_cache
	@echo "Clean done."
