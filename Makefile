# Every target runs inside the container, so the reviewer needs nothing on the
# host but Docker. Host Python is irrelevant by design.

COMPOSE := docker compose
RUN     := $(COMPOSE) run --rm app
CLI     := python -m src.cli

.DEFAULT_GOAL := help
.PHONY: help up down reset psql logs shell db-check ingest index benchmark-dense benchmark-sparse benchmark-hybrid benchmark-rerank benchmark-rag ask-rag tune evaluate benchmark report pipeline test lint fmt typecheck clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up:  ## Start PostgreSQL + pgvector and wait until healthy
	$(COMPOSE) up -d db
	@printf "waiting for db"
	@for i in $$(seq 1 60); do \
		status=$$($(COMPOSE) ps -q db | xargs docker inspect -f '{{.State.Health.Status}}' 2>/dev/null); \
		if [ "$$status" = "healthy" ]; then printf " ready\n"; exit 0; fi; \
		if [ "$$status" = "unhealthy" ]; then printf " failed\n"; $(COMPOSE) logs --tail 50 db; exit 1; fi; \
		printf "."; sleep 2; \
	done; \
	printf " timed out\n"; echo "db did not become healthy within 120s"; $(COMPOSE) logs --tail 50 db; exit 1

down:  ## Stop containers (keeps the data volume)
	$(COMPOSE) down

reset:  ## Stop containers and DELETE the data volume
	$(COMPOSE) down -v

psql:  ## Open a psql shell on the database
	$(COMPOSE) exec db psql -U $${APEX_PG_USER:-apex} -d $${APEX_PG_DATABASE:-apex_rag}

logs:  ## Tail the database logs
	$(COMPOSE) logs -f db

shell:  ## Interactive shell inside the app container
	$(RUN) bash

db-check:  ## Verify PostgreSQL, pgvector, the schema and the HNSW index
	$(RUN) $(CLI) db-check

ingest:  ## Phase 3 -- load, chunk, embed and store the knowledge base
	$(RUN) $(CLI) ingest

index:  ## PLANNED -- CLI "index" command is not implemented yet
	$(RUN) $(CLI) index

benchmark-dense:  ## Phase 4C -- exact vs natural vs forced-HNSW dense benchmark
	$(RUN) python scripts/benchmark_dense.py --top-k 10 --ef-search 10 40 100 --repetitions 5

benchmark-sparse:  ## Phase 5A -- sparse FTS benchmark and Chinese probe
	$(RUN) python scripts/benchmark_sparse.py --top-k 10 --repetitions 5

benchmark-hybrid:  ## Phase 6B -- five-arm dense/sparse/union/RRF evaluation
	$(RUN) python scripts/benchmark_hybrid.py --top-k 10 --candidate-k 10 --rrf-k 60 --repetitions 5 --warmup-passes 1

benchmark-rerank:  ## Phase 7B -- historical RRF Top-10 + cross-encoder reranker
	$(RUN) python scripts/benchmark_rerank.py --top-k 10 --candidate-k 10 --rrf-k 60 --batch-size 16 --activation identity

benchmark-rag:  ## Phase 8 -- RAG QA evaluation (mock LLM; historical RRF replay)
	$(RUN) python scripts/benchmark_rag.py --provider mock --retrieval-mode historical

ask-rag:  ## Phase 8 -- ask one question (mock LLM; pass Q="..." to override)
	$(RUN) python scripts/ask_rag.py "$(if $(Q),$(Q),What is the foreign transaction fee?)" --mock --retrieval-mode historical

tune:  ## PLANNED -- HNSW sweep CLI is not implemented yet
	$(RUN) $(CLI) benchmark --mode index

evaluate:  ## Alias for benchmark-hybrid (Phase 6B)
	$(MAKE) --no-print-directory benchmark-hybrid

report:  ## PLANNED -- Phase 7 report CLI is not implemented yet
	$(RUN) $(CLI) report

pipeline:  ## PLANNED -- end-to-end pipeline CLI is not implemented yet
	$(RUN) $(CLI) pipeline

test:  ## Run the test suite
	$(RUN) pytest

lint:  ## Lint with ruff
	$(RUN) ruff check src tests

fmt:  ## Auto-fix lint issues and format
	$(RUN) ruff check --fix src tests
	$(RUN) ruff format src tests

typecheck:  ## Static type checking
	$(RUN) mypy src

clean:  ## Remove caches and generated reports (keeps the data volume)
	$(RUN) find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf reports/figures/*.png reports/results.md .mypy_cache .ruff_cache .pytest_cache
