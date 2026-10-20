# Project targets run inside the container, so the reviewer needs nothing on the
# host but Docker. The one exception is `clean`: it only removes host-side Python
# caches, deliberately avoiding `docker compose run` (which would start the db).
#
# Extra flags: every target that runs a benchmark or the CLI forwards `ARGS`,
# e.g. `make tune ARGS="--synthetic-scale 100 --overwrite"`.

COMPOSE := docker compose
RUN     := $(COMPOSE) run --rm app
CLI     := python -m src.cli
ARGS    ?=

.DEFAULT_GOAL := help

.PHONY: help up down reset psql logs shell db-check ingest index-build index-list tune \
	benchmark-dense benchmark-sparse benchmark-hybrid benchmark-rerank benchmark-rag \
	benchmark-retrieval benchmark-index benchmark-chunking benchmark-fusion benchmark-all \
	judge ask-rag evaluate report pipeline warm-cache api acceptance verify-container \
	test lint fmt typecheck clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

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

db-check:  ## Verify PostgreSQL, pgvector, the schema and the vector indexes
	$(RUN) $(CLI) db-check

ingest:  ## Load, chunk, embed and store the knowledge base (ARGS="--strategy structured")
	$(RUN) $(CLI) ingest $(ARGS)

index-build:  ## Build one shadow index (ARGS="--method ivfflat --lists 8")
	$(RUN) $(CLI) index build $(ARGS)

index-list:  ## List the HNSW / IVFFlat indexes on chunks
	$(RUN) $(CLI) index list

tune:  ## HNSW / IVFFlat parameter sweep (ARGS="--synthetic-scale 100")
	$(RUN) $(CLI) index tune $(ARGS)

benchmark-dense:  ## Dense: exact vs natural vs forced-HNSW
	$(RUN) python scripts/benchmark_dense.py --top-k 10 --ef-search 10 40 100 --repetitions 5 $(ARGS)

benchmark-sparse:  ## Sparse FTS benchmark and Chinese probe
	$(RUN) python scripts/benchmark_sparse.py --top-k 10 --repetitions 5 $(ARGS)

benchmark-hybrid:  ## Five-arm dense/sparse/union/RRF evaluation
	$(RUN) python scripts/benchmark_hybrid.py --top-k 10 --candidate-k 10 --rrf-k 60 --repetitions 5 --warmup-passes 1 $(ARGS)

benchmark-rerank:  ## RRF Top-10 + cross-encoder reranker
	$(RUN) python scripts/benchmark_rerank.py --top-k 10 --candidate-k 10 --rrf-k 60 --batch-size 16 --activation identity $(ARGS)

benchmark-rag:  ## End-to-end RAG QA (mock LLM; historical RRF replay)
	$(RUN) python scripts/benchmark_rag.py --provider mock --retrieval-mode historical $(ARGS)

judge:  ## LLM-as-a-Judge smoke (mock by default; ARGS="--provider real --judge-provider real --limit 3")
	$(RUN) $(CLI) evaluate-judge $(ARGS)

benchmark-retrieval:  ## Unified six-arm retrieval evaluation
	$(RUN) python scripts/benchmark_retrieval.py --top-k 10 $(ARGS)

benchmark-index:  ## HNSW m / ef_construction and IVFFlat lists / probes sweep
	$(RUN) python scripts/benchmark_index.py $(ARGS)

benchmark-chunking:  ## document vs fixed_window vs structured
	$(RUN) python scripts/benchmark_chunking.py $(ARGS)

benchmark-fusion:  ## RRF k and candidate-pool sensitivity
	$(RUN) python scripts/benchmark_fusion.py $(ARGS)

benchmark-all:  ## Retrieval + index + chunking into reports/unified/
	$(RUN) python scripts/benchmark_all.py $(ARGS)

ask-rag:  ## Ask one question (mock LLM by default; Q="..." overrides)
	$(RUN) python scripts/ask_rag.py "$(if $(Q),$(Q),What is the foreign transaction fee?)" --mock --retrieval-mode historical

evaluate:  ## Print the metrics recorded in reports/unified/benchmark.json
	$(RUN) $(CLI) evaluate $(ARGS)

report:  ## Assemble reports/final_report.md and the figures
	$(RUN) $(CLI) report $(ARGS)

pipeline:  ## Run retrieval + index + chunking, then build the report
	$(RUN) $(CLI) pipeline $(ARGS)

warm-cache:  ## Download the embedding and reranker models into the hf_cache volume
	$(COMPOSE) run --rm --no-deps app python scripts/warm_cache.py $(ARGS)

api:  ## Start the read-only FastAPI service on http://localhost:8000
	$(COMPOSE) --profile api up -d api

acceptance: api  ## Start the read-only acceptance dashboard at /acceptance/
	@printf "Acceptance dashboard: http://localhost:8000/acceptance/\n"

verify-container:  ## Run the full verification sequence and log it to reports/container/
	@mkdir -p reports/container
	$(MAKE) --no-print-directory db-check 2>&1 | tee reports/container/db-check.log
	$(MAKE) --no-print-directory benchmark-all ARGS="--overwrite" 2>&1 | tee reports/container/benchmark-all.log
	$(MAKE) --no-print-directory report ARGS="--overwrite" 2>&1 | tee reports/container/report.log

test:  ## Run the test suite
	$(RUN) pytest $(ARGS)

lint:  ## Lint with ruff
	$(RUN) ruff check src scripts tests

fmt:  ## Auto-fix lint issues and format
	$(RUN) ruff check --fix src scripts tests

typecheck:  ## Static type checking
	$(RUN) mypy src

clean:  ## Remove host Python caches (keeps reports and the data volume)
	# Delivered reports (reports/*.md, reports/*.json, reports/figures/) are
	# intentionally not touched. This target also avoids `docker compose run`,
	# which would start the database dependency just to delete local caches.
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache .pytest_cache
