PYTHON = python
BUILD_DIR = ./

# Style
style:
	black .
	flake8 ${BUILD_DIR}
	${PYTHON} -m isort ${BUILD_DIR}
