PYTHON ?= python

.PHONY: test
test:
	 $(PYTHON) -m unittest discover -p '*_test.py'
