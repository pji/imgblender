.PHONY: build
build:
	python -m pipenv install --dev -e .
	sphinx-build -b html docs/source/ docs/build/html
	python -m build
	twine check dist/*

.PHONY: buildt
buildt:
	python -m pipenv install --dev -e .

.PHONY: clean
clean:
	rm -rf docs/build/html
	rm -rf dist
	rm -rf src/imgblender.egg-info
	rm -rf tests/__pycache__
	rm -rf tests/*.pyc
	rm -rf src/imgblender/__pycache__
	rm -rf src/imgblender/*.pyc
	rm -rf src/imgblender/pattern/__pycache__
	rm -f *.log
	rm -f *.json
	rm -f *.jpg
	python -m pipenv uninstall imgblender
	python -m pipenv install --dev -e .

.PHONY: docs
docs:
	python examples/build_doc_images.py
	rm -rf docs/build/html
	sphinx-build -b html docs/source/ docs/build/html

.PHONY: pre
pre:
	python precommit.py
	git status

.PHONY: test
test:
	python -m pytest --capture=fd


.PHONY: testv
testv:
	python -m pytest -vv  --capture=fd
