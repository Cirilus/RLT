up:
	poetry run uvicorn app:app --reload --port 8000

.PHONY: migrate-revision
migrate-revision:
	poetry run alembic revision --autogenerate -m $(name)

.PHONY: migrate-up
migrate-up:
	poetry run alembic upgrade $(rev)

.PHONY: test
test:
	poetry run pytest

.PHONY: lint
lint:
	poetry run pylint .