# Repo-level convenience targets
solver:
	$(MAKE) -C cfd_engine

cuda:
	$(MAKE) -C cfd_engine cuda

clean:
	$(MAKE) -C cfd_engine clean

test: solver
	pytest -m "not slow"

test-all: solver
	pytest

.PHONY: solver cuda clean test test-all
