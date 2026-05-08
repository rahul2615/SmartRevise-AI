from app import app, db, bcrypt
from models import User

with app.app_context():
    db.create_all()
    admin = User.query.filter_by(role='admin').first()
    if not admin:
        hashed = bcrypt.generate_password_hash('admin123').decode('utf-8')
        admin = User(username='Admin', email='admin@smartrevise.com', password=hashed, role='admin')
        db.session.add(admin)
        db.session.commit()
        print("Admin user created successfully.")
    else:
        print("Admin user already exists.")
