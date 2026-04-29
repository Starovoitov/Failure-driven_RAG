POETRY ?= poetry

.PHONY: install lint fix

install:
	$(POETRY) install

lint:
	$(POETRY) run black --check .
	$(POETRY) run ruff check .

fix:
	$(POETRY) run black .
	$(POETRY) run ruff check --fix .
