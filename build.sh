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
python -c "from app import app, db, User, bcrypt; \
app.app_context().push(); \
db.create_all(); \
admin = User.query.filter_by(role='admin').first(); \
if not admin: \
    db.session.add(User(username='Admin', email='admin@smartrevise.com', password=bcrypt.generate_password_hash('admin123').decode('utf-8'), role='admin')); \
    db.session.commit()"

echo "Running Database Migrations..."
# Initialize DB if using Postgres, else handled by Flask
flask db upgrade || true

echo "Build complete."
