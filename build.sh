#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading Spacy English Model..."
python -m spacy download en_core_web_sm

echo "Downloading NLTK Data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

echo "Initializing Database Tables..."
python init_db.py

echo "Running Database Migrations..."
# Initialize DB if using Postgres, else handled by Flask
flask db upgrade || true

echo "Build complete."
