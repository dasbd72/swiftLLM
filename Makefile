lint:
	isort --line-length 80 --profile google .
	pyink --line-length 80 --preview --unstable .
	flake8 --ignore=E203,E501,W503 --exclude venv
