# Makefile

# Shell
SHELL := /bin/bash
#Project
PROJECT = "image_pro"
#Version
RELEASE = "0.0.1"
# Path to source directory
PROJECTPATH := .
# OS machine
OS_TYPE = 'uname -a'
# CPP EXTENSIONS
CPP_EXT = ".cpp"

# SETUP
SETUPFILE = "setup.py"

# VENV

_virtualenv:
	# create virtual env
	virtualenv _virtualenv
	_virtualenv/bin/pip install --upgrade pip
	_virtualenv/bin/pip install --upgrade setuptools

_use_env:
	# use virtual env
	source _virtualenv/bin/activate

##### PYTHON

prebuild:
	@echo "Preparing build"
	@echo "Installing requirements"
	pip install -r requirements.txt

setup:
	echo "Setup..."
	sudo python setup.py install

test:
	echo "Test"
	python setup.py test

clean:
	rm -f MANIFEST
	rm -rf build dist

deactivate_env:
	deactivate


all: prebuild setup

.PHONY: clean