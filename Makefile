# Repo-level convenience targets
solver:
	$(MAKE) -C cfd_engine

clean:
	$(MAKE) -C cfd_engine clean

test: solver
	pytest -m "not slow"

test-all: solver
	pytest

.PHONY: solver clean test test-all
