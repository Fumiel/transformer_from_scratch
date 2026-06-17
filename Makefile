.PHONY: setup test lint build-cpp run-cpp clean

setup:
	python -m pip install -U pip
	python -m pip install -e .[dev]

test:
	pytest

lint:
	ruff check python tests

build-cpp:
	cmake -S . -B build
	cmake --build build

run-cpp: build-cpp
	./build/tiny_transformer_cpp

clean:
	rm -rf build .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
