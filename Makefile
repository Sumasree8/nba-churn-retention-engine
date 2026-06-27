# NBA Churn Retention Engine — Makefile
# Usage: make <target>

.PHONY: help install data pipeline api dashboard test lint format docker clean

PYTHON := python
PIP := pip

help:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  NBA Churn Retention Engine — Available Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make install       Install all dependencies"
	@echo "  make data          Generate synthetic dataset (50K records)"
	@echo "  make pipeline      Run full end-to-end pipeline (synthetic data)"
	@echo "  make pipeline-db   Run pipeline against a real database"
	@echo "  make pipeline-tune Run pipeline with hyperparameter tuning"
	@echo "  make api           Start FastAPI server (port 8000)"
	@echo "  make dashboard     Launch Streamlit dashboard (port 8501)"
	@echo "  make test          Run all tests with coverage"
	@echo "  make test-unit     Run unit tests only"
	@echo "  make test-data     Run data validation tests only"
	@echo "  make lint          Run flake8 linter"
	@echo "  make format        Auto-format with black"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-up     Start all services with docker-compose"
	@echo "  make docker-down   Stop all services"
	@echo "  make mlflow        Start MLflow tracking server"
	@echo "  make clean         Remove generated artifacts"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt -r requirements-ml.txt
	@echo "✓ Dependencies installed"

install-dashboard:
	$(PIP) install -r requirements.txt
	@echo "✓ Dashboard-only dependencies installed"

data:
	$(PYTHON) src/data/generate_data.py --n 50000 --output data/raw/saas_churn_dataset.csv
	@echo "✓ Dataset generated: data/raw/saas_churn_dataset.csv"

pipeline:
	$(PYTHON) src/pipeline.py --n 50000
	@echo "✓ Pipeline complete. Check data/sample_outputs/nba_recommendations.csv"

pipeline-tune:
	$(PYTHON) src/pipeline.py --n 50000 --tune
	@echo "✓ Pipeline with tuning complete."

pipeline-db:
	PYTHONPATH=. $(PYTHON) src/pipeline.py --source db
	@echo "✓ Pipeline complete using production database (config/churn_query.sql)."

pipeline-quick:
	$(PYTHON) src/pipeline.py --n 10000
	@echo "✓ Quick pipeline complete."

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
	
dashboard:
	streamlit run dashboards/streamlit/app.py --server.port 8501

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "✓ Tests complete. Coverage report: htmlcov/index.html"

test-unit:
	pytest tests/unit/ -v

test-data:
	pytest tests/data_validation/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	flake8 src/ api/ tests/
	isort --check-only src/ api/ tests/
	black --check src/ api/ tests/ --line-length 100

format:
	isort src/ api/ tests/
	black src/ api/ tests/ --line-length 100

mlflow:
	mlflow server --host 0.0.0.0 --port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns
	@echo "MLflow UI: http://localhost:5000"

docker-build:
	docker build -t nba-churn-engine:latest -f deployment/docker/Dockerfile .

docker-up:
	cd deployment/docker && docker compose up --build -d
	@echo "✓ Services started:"
	@echo "  API:       http://localhost:8000"
	@echo "  Dashboard: http://localhost:8501"
	@echo "  MLflow:    http://localhost:5000"

docker-down:
	cd deployment/docker && docker compose down

docker-logs:
	cd deployment/docker && docker compose logs -f

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f coverage.xml
	rm -rf htmlcov/
	@echo "✓ Cleaned up temporary files"

clean-data:
	rm -f data/raw/*.csv data/processed/*.csv data/sample_outputs/*.csv
	rm -f models/artifacts/*.pkl models/artifacts/*.json
	@echo "✓ Data and model artifacts removed"
