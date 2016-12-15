SRC_PATH=./src

default: run

run:
	python "${SRC_PATH}/main.py"

lint:
	flake8 "${SRC_PATH}" --exclude=.tox