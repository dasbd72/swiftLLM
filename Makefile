C_CXX_SOURCES := $(shell find csrc -name *.c -or -name *.cpp -or -name *.cc -or -name *.h -or -name *.hpp)

lint:
	isort --line-length 80 --profile google .
	pyink --line-length 80 --preview --unstable .
	flake8 --ignore=E203,E501,W503 --exclude venv
	clang-format -i $(C_CXX_SOURCES)
