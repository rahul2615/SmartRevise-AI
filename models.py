from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='student')  # 'admin' or 'student'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Real-time Metrics
    last_active_date = db.Column(db.Date, default=None)
    current_streak = db.Column(db.Integer, default=0)
    total_study_minutes = db.Column(db.Integer, default=0)

    # Session Security
    session_token = db.Column(db.String(100), nullable=True)   # Single-session enforcement
    failed_login_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime, nullable=True)
    last_login_at = db.Column(db.DateTime, nullable=True)
    last_login_ip = db.Column(db.String(50), nullable=True)

    @property
    def is_admin(self):
        return self.role == 'admin'

    @property
    def is_locked(self):
        """Check if the account is currently locked out."""
        if self.locked_until and self.locked_until > datetime.utcnow():
            return True
        return False

    def __repr__(self):
        return f'<User {self.username} ({self.role})>'

class RevisionNote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    topic = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    phase = db.Column(db.String(50), default='Phase 1')
    tags = db.Column(db.String(200), default='')
    is_pinned = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MCQScore(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    topic = db.Column(db.String(200), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    total_questions = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class StudyPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    topics = db.Column(db.Text, nullable=False) # JSON string or comma-separated
    completed = db.Column(db.Boolean, default=False)

class CodeSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    language = db.Column(db.String(50), nullable=False)
    code = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), nullable=False) # 'Success', 'Error'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class StudyRoadmap(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    syllabus_text = db.Column(db.Text, nullable=False)
    roadmap_json = db.Column(db.Text, nullable=False) # JSON string of steps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class PDFDocument(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(300), nullable=False)
    extracted_text = db.Column(db.Text, nullable=False)
    num_pages = db.Column(db.Integer, default=0)
    word_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ==========================================
# Study Groups - Real-Time Collaboration
# ==========================================
class StudyGroup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, default='')
    invite_code = db.Column(db.String(8), unique=True, nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    shared_pdf_id = db.Column(db.Integer, db.ForeignKey('pdf_document.id'), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    members = db.relationship('GroupMember', backref='group', lazy=True, cascade='all, delete-orphan')
    messages = db.relationship('GroupMessage', backref='group', lazy=True, cascade='all, delete-orphan')

class GroupMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    group_id = db.Column(db.Integer, db.ForeignKey('study_group.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    role = db.Column(db.String(20), default='member')  # 'admin', 'member'
    joined_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='group_memberships')

class GroupMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    group_id = db.Column(db.Integer, db.ForeignKey('study_group.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    msg_type = db.Column(db.String(20), default='text')  # 'text', 'annotation', 'quiz_result', 'system'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User')

# ==========================================
# AI Tutor - Persistent Chat History
# ==========================================
class ChatConversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), default='New Chat')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = db.relationship('ChatMessage', backref='conversation', lazy=True,
                               cascade='all, delete-orphan', order_by='ChatMessage.created_at')

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('chat_conversation.id'), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # 'user' or 'ai'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ==========================================
# Admin Audit Log
# ==========================================
class AdminAuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action = db.Column(db.String(50), nullable=False)  # 'role_change', 'delete_user', 'reset_password'
    target_user_id = db.Column(db.Integer, nullable=True)
    details = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    admin = db.relationship('User', foreign_keys=[admin_id])

