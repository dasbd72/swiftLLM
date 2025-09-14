C_CXX_SOURCES := $(shell find csrc -name *.c -or -name *.cpp -or -name *.cc -or -name *.h -or -name *.hpp)

help:
	@echo "lint - Run isort, black and flake8."

lint:
	isort --line-length 79 --profile black .
	black --line-length 79 --preview --enable-unstable-feature string_processing .
	flake8 --ignore=E203,E501,W503 --exclude venv
	clang-format -i $(C_CXX_SOURCES)

install_swiftllm_c:
	pip install -e csrc --no-build-isolation -v
