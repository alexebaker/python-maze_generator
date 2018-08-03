all: docs

.PHONY: clean
clean:
	find . -name "*.pyc" -delete

.PHONY: docs
docs:
	python setup.py build_sphinx --build-dir docs/build
