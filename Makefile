.PHONY: install install-dev lint test backtest report dataset loadtest demo demo-down docker-build docker-up docker-down deploy clean

PYTHON ?= python3
IMAGE_NAME ?= chatbot-platform
IMAGE_TAG ?= latest
HELM_RELEASE ?= chatbot
K8S_NAMESPACE ?= default

install:
	$(PYTHON) -m pip install . --break-system-packages

install-dev:
	$(PYTHON) -m pip install ".[dev]" --break-system-packages
	pre-commit install

lint:
	ruff check src tests evaluation
	ruff format --check src tests evaluation
	vulture src --min-confidence 80

format:
	ruff check --fix src tests evaluation
	ruff format src tests evaluation

test:
	LLM_MOCK_MODE=true pytest tests/ -v --tb=short

backtest:
	LLM_MOCK_MODE=true $(PYTHON) -m evaluation.backtest evaluation/sample_conversations.jsonl

report:
	LLM_MOCK_MODE=true $(PYTHON) -m evaluation.report evaluation/sample_conversations.jsonl evaluation/report.html
	@echo "Open evaluation/report.html in your browser"

dataset:
	$(PYTHON) -m evaluation.generate_dataset evaluation/sample_conversations.jsonl

loadtest:
	locust -f evaluation/locustfile.py --host http://localhost:8000 --headless -u 50 -r 5 --run-time 1m --csv evaluation/loadtest

docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down -v

demo:
	docker compose -f demo/docker-compose.yml up --build -d
	@echo ""
	@echo "  Chat UI:    http://localhost:8080"
	@echo "  API docs:   http://localhost:8000/docs"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"
	@echo "  Prometheus: http://localhost:9091"
	@echo ""
	@echo "Try: curl -s http://localhost:8000/chat/strict -H 'Content-Type: application/json' -d '{\"message\": \"What is Python?\"}' | python3 -m json.tool"

demo-down:
	docker compose -f demo/docker-compose.yml down -v

deploy:
	helm upgrade --install $(HELM_RELEASE) deploy/helm/chatbot \
		--namespace $(K8S_NAMESPACE) \
		--create-namespace \
		--set image.repository=$(IMAGE_NAME) \
		--set image.tag=$(IMAGE_TAG)

deploy-canary:
	helm upgrade --install $(HELM_RELEASE)-canary deploy/helm/chatbot \
		--namespace $(K8S_NAMESPACE) \
		--set replicaCount=1 \
		--set image.repository=$(IMAGE_NAME) \
		--set image.tag=$(IMAGE_TAG)-canary \
		--set autoscaling.enabled=false

run:
	LLM_MOCK_MODE=true uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

drift-check:
	$(PYTHON) -m src.drift.alert_cronjob

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache data/feedback.db data/drift_events.db evaluation/report.html
