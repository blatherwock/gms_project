SRC_PATH=./src

default: run

run:
	python3 "${SRC_PATH}/main.py"

lint:
	flake8 "${SRC_PATH}" --exclude=.tox