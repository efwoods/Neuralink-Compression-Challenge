#!/bin/sh

# Update the encode and decode files
cp src/brainwire/encode.py encode
cp src/brainwire/decode.py decode

# Update the pypi package repository
rm -rf dist/*
python -m build
python -m twine upload --repository pypi dist/*
