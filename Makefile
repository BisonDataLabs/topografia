.PHONY: setup run lint test check clean

setup:
	python -m pip install -r requirements-dev.txt

run:
	python -m streamlit run app.py

lint:
	python -m ruff format --check app.py universal_app.py topografia tests
	python -m ruff check app.py universal_app.py topografia tests

test:
	python -m pytest

check: lint test
	python scripts/check_secrets.py

clean:
	python -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', 'htmlcov']]"
