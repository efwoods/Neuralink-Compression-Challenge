#!/bin/sh

# Update the encode and decode files
cp encode src/brainwire/encode.py
cp decode src/brainwire/decode.py

# Update the pypi package repository
rm -rf dist/*
python -m build
python -m twine upload --repository pypi dist/*
