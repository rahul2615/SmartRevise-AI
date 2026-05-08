from app import app, db, bcrypt
from models import User
from flask_migrate import stamp
from sqlalchemy import inspect

with app.app_context():
    inspector = inspect(db.engine)
    needs_stamp = not inspector.has_table("alembic_version")
    
    db.create_all()
    
    if needs_stamp:
        # Stamp the migration to head so flask db upgrade doesn't run the initial migration
        stamp()
        print("Stamped database migrations to head.")
        
    admin = User.query.filter_by(role='admin').first()
    if not admin:
        hashed = bcrypt.generate_password_hash('admin123').decode('utf-8')
        admin = User(username='Admin', email='admin@smartrevise.com', password=hashed, role='admin')
        db.session.add(admin)
        db.session.commit()
        print("Admin user created successfully.")
    else:
        print("Admin user already exists.")
