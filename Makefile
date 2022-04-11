export ROOT_DIR=${PWD}
VENV_NAME?=venv
PYTHON=${VENV_NAME}/bin/python

process_data:
	echo "Process Raw Data"
	cd data/raw; echo "I'm in dir"; find . -name rpe.json | cpio -pdm ../processed/
	$(PYTHON) process_raw_data.py
