.PHONY: install run_builder run_inference clean test

install:
	poetry install --no-root --with dev

run_builder:
	cd src && poetry run python runner_builder.py

run_inference:
	cd src && poetry run python runner_inference.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf src/logs/*.log*
	rm -rf src/model/models/*
	find . -type d -name ".pytest_cache" -delete
