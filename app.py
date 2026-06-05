from flask import Flask, render_template, redirect, url_for, flash, request, jsonify, abort, session
from models import db, User, RevisionNote, MCQScore, StudyPlan, CodeSubmission, StudyRoadmap, PDFDocument, StudyGroup, GroupMember, GroupMessage, ChatConversation, ChatMessage, AdminAuditLog
import requests
import json
import random
import string
import os
import uuid
import hashlib
from functools import wraps
from datetime import datetime, timedelta
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect
from authlib.integrations.flask_client import OAuth
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import re

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if it exists

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# ==========================================
# OAuth Setup
# ==========================================
oauth = OAuth(app)
oauth.register(
    name='google',
    client_id=os.environ.get('GOOGLE_CLIENT_ID', 'placeholder_client_id'),
    client_secret=os.environ.get('GOOGLE_CLIENT_SECRET', 'placeholder_client_secret'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

# ==========================================
# Security Configuration
# ==========================================
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(32).hex())
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///database_v2.db')
if app.config['SQLALCHEMY_DATABASE_URI'].startswith('postgres://'):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# PASTE YOUR GEMINI API KEY HERE
app.config['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY', 'AIzaSyAdvTFyqGfq1HmixvmlmhFYxhsMrnNDOT0')

# Session Cookie Hardening
app.config['SESSION_COOKIE_HTTPONLY'] = True      # Prevent JavaScript access to session cookie
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'     # Prevent cross-site request attacks
app.config['SESSION_COOKIE_NAME'] = 'sr_session'  # Custom cookie name
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)  # Session expires in 2 hours
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=7)     # Remember me duration
app.config['REMEMBER_COOKIE_HTTPONLY'] = True
app.config['REMEMBER_COOKIE_SAMESITE'] = 'Lax'

# Security Constants
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION = timedelta(minutes=15)
IDLE_TIMEOUT = timedelta(minutes=30)
ADMIN_REAUTH_TIMEOUT = timedelta(minutes=10)

db.init_app(app)
migrate = Migrate(app, db)  # Flask-Migrate / Alembic for schema migrations
bcrypt = Bcrypt(app)
csrf = CSRFProtect(app)     # CSRF protection on all forms
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'warning'
login_manager.session_protection = 'basic'  # 'basic' is safer behind reverse proxies (Render, etc.)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# Rate Limiter — prevent API abuse
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# ==========================================
# Security Helper Functions
# ==========================================
def _generate_session_fingerprint():
    """Generate a fingerprint from the user's browser User-Agent.
    
    Note: IP is intentionally excluded because reverse proxies (Render,
    Cloudflare, etc.) can report different IPs across consecutive requests,
    which would cause false-positive session invalidations.
    """
    ua = request.headers.get('User-Agent', '')
    return hashlib.sha256(ua.encode()).hexdigest()

def _log_admin_action(action, target_user_id=None, details=None):
    """Record an admin action in the audit log."""
    log = AdminAuditLog(
        admin_id=current_user.id,
        action=action,
        target_user_id=target_user_id,
        details=details,
        ip_address=request.remote_addr
    )
    db.session.add(log)

# ==========================================
# Role-Based Access Control Decorators
# ==========================================
def admin_required(f):
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if not current_user.is_admin:
            abort(403)
        # Check admin re-authentication
        admin_authed_at = session.get('admin_authed_at')
        if not admin_authed_at or datetime.utcnow() - datetime.fromisoformat(admin_authed_at) > ADMIN_REAUTH_TIMEOUT:
            session['admin_next'] = request.url
            flash('Please re-enter your password to access the admin panel.', 'warning')
            return redirect(url_for('admin_auth'))
        return f(*args, **kwargs)
    return decorated

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==========================================
# Session Security Middleware
# ==========================================
@app.before_request
def enforce_session_security():
    """Validate session on every request: single-session, fingerprint, idle timeout."""
    # Skip for static files and non-authenticated users
    if request.endpoint == 'static' or not current_user.is_authenticated:
        return

    # 1. Single-Session Enforcement — check token matches DB
    session_token = session.get('session_token')
    if not session_token or session_token != current_user.session_token:
        logout_user()
        session.clear()
        flash('Session expired. You may have logged in from another device or browser.', 'warning')
        return redirect(url_for('login'))

    # 2. Session Fingerprint Validation — detect session hijacking
    stored_fingerprint = session.get('session_fingerprint')
    current_fingerprint = _generate_session_fingerprint()
    if stored_fingerprint and stored_fingerprint != current_fingerprint:
        # Invalidate the session token in DB too
        current_user.session_token = None
        db.session.commit()
        logout_user()
        session.clear()
        flash('Session invalidated due to suspicious activity. Please log in again.', 'danger')
        return redirect(url_for('login'))

    # 3. Idle Timeout — auto-logout after 30 min inactivity
    last_activity = session.get('last_activity')
    if last_activity:
        last_dt = datetime.fromisoformat(last_activity)
        if datetime.utcnow() - last_dt > IDLE_TIMEOUT:
            current_user.session_token = None
            db.session.commit()
            logout_user()
            session.clear()
            flash('You have been logged out due to inactivity.', 'info')
            return redirect(url_for('login'))

    # Update last activity timestamp
    session['last_activity'] = datetime.utcnow().isoformat()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember_me') == 'on'

        user = User.query.filter_by(email=email).first()

        if user and user.is_active is False:
            flash('This account has been deactivated by an administrator.', 'danger')
            return render_template('login.html')

        # Check account lockout
        if user and user.is_locked:
            remaining = int((user.locked_until - datetime.utcnow()).total_seconds() / 60) + 1
            flash(f'Account temporarily locked. Try again in {remaining} minutes.', 'danger')
            return render_template('login.html')

        if user and bcrypt.check_password_hash(user.password, password):
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login_at = datetime.utcnow()
            user.last_login_ip = request.remote_addr

            # Generate single-session token
            new_token = str(uuid.uuid4())
            user.session_token = new_token
            db.session.commit()

            login_user(user, remember=remember)
            session.permanent = True
            session['session_token'] = new_token
            session['session_fingerprint'] = _generate_session_fingerprint()
            session['last_activity'] = datetime.utcnow().isoformat()

            flash('Login Successful!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            # Track failed login attempts
            if user:
                user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
                if user.failed_login_attempts >= MAX_LOGIN_ATTEMPTS:
                    user.locked_until = datetime.utcnow() + LOCKOUT_DURATION
                    flash(f'Too many failed attempts. Account locked for {int(LOCKOUT_DURATION.total_seconds() / 60)} minutes.', 'danger')
                else:
                    remaining = MAX_LOGIN_ATTEMPTS - user.failed_login_attempts
                    flash(f'Invalid credentials. {remaining} attempt(s) remaining.', 'danger')
                db.session.commit()
            else:
                flash('Invalid credentials. Please check email and password.', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def register():
    if request.method == 'POST':
        username = request.form.get('fullname') # Matching the name in register.html
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        # 1. Email Domain Validation
        if not email.endswith('@gmail.com') and not email.endswith('@smartrevise.com'):
            flash('Registration failed: Email must end with @gmail.com or @smartrevise.com', 'danger')
            return render_template('register.html')

        # 2. Password Complexity Validation
        if len(password) < 8 or not re.search(r'[A-Z]', password) or not re.search(r'\d', password):
            flash('Registration failed: Password must be at least 8 characters long and contain at least one uppercase letter and one number.', 'danger')
            return render_template('register.html')
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_password)
        
        try:
            db.session.add(user)
            db.session.commit()
            flash('Account created! You can now login', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Error: {e}', 'danger')
            
    return render_template('register.html')

# ==========================================
# Google OAuth Routes
# ==========================================
@app.route('/login/google')
def login_google():
    redirect_uri = url_for('authorize_google', _external=True)
    if redirect_uri.startswith('http://') and not ('localhost' in request.host or '127.0.0.1' in request.host):
        redirect_uri = redirect_uri.replace('http://', 'https://', 1)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route('/login/google/authorize')
def authorize_google():
    try:
        token = oauth.google.authorize_access_token()
        resp = oauth.google.get('https://www.googleapis.com/oauth2/v3/userinfo')
        user_info = resp.json()
        
        email = user_info.get('email', '').lower()
        
        # Domain validation for OAuth too
        if not email.endswith('@gmail.com') and not email.endswith('@smartrevise.com'):
            flash('Login failed: Only @gmail.com and @smartrevise.com domains are allowed.', 'danger')
            return redirect(url_for('login'))
            
        user = User.query.filter_by(email=email).first()
        
        if user and user.is_active is False:
            flash('This account has been deactivated by an administrator.', 'danger')
            return redirect(url_for('login'))
        
        if not user:
            # Auto-register Google users
            username = user_info.get('name', email.split('@')[0])
            # Use a random secure password for OAuth accounts
            random_pw = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            hashed_password = bcrypt.generate_password_hash(random_pw).decode('utf-8')
            
            user = User(username=username, email=email, password=hashed_password)
            db.session.add(user)
            db.session.commit()
            
        # Check lockout
        if user.is_locked:
            remaining = int((user.locked_until - datetime.utcnow()).total_seconds() / 60) + 1
            flash(f'Account temporarily locked. Try again in {remaining} minutes.', 'danger')
            return redirect(url_for('login'))
            
        # Log the user in securely
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = datetime.utcnow()
        user.last_login_ip = request.remote_addr
        
        new_token = str(uuid.uuid4())
        user.session_token = new_token
        db.session.commit()
        
        login_user(user, remember=True)
        session.permanent = True
        session['session_token'] = new_token
        session['session_fingerprint'] = _generate_session_fingerprint()
        session['last_activity'] = datetime.utcnow().isoformat()
        
        flash('Successfully logged in with Google!', 'success')
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        flash(f'Google Login Failed: {str(e)}', 'danger')
        return redirect(url_for('login'))

@app.before_request
def update_streak():
    if current_user.is_authenticated:
        today = datetime.utcnow().date()
        last_active = current_user.last_active_date
        
        if last_active != today:
            if last_active == today - timedelta(days=1):
                # Consecutive day
                current_user.current_streak = (current_user.current_streak or 0) + 1
            else:
                # Broken streak or first time
                current_user.current_streak = 1
                
            current_user.last_active_date = today
            db.session.commit()

@app.route('/api/update_time', methods=['POST'])
@login_required
def update_time():
    current_user.total_study_minutes = (current_user.total_study_minutes or 0) + 1
    db.session.commit()
    return jsonify({'success': True, 'total': current_user.total_study_minutes})

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = current_user.id
    
    # 1. Basic Stats
    notes_count = RevisionNote.query.filter_by(user_id=user_id).count()
    scores = MCQScore.query.filter_by(user_id=user_id).all()
    problems_solved = notes_count + len(scores) * 10 
    
    avg_score = 0
    if scores:
        total_correct = 0
        total_questions_all = 0
        for s in scores:
            # s.score is percentage, recover raw correct count
            correct = (s.score / 100) * s.total_questions
            total_correct += correct
            total_questions_all += s.total_questions
            
        if total_questions_all > 0:
            avg_score = int((total_correct / total_questions_all) * 100)

    # 2. Real-time Streak & Time
    streak = current_user.current_streak or 0
    
    # Calculate hours for display
    total_minutes = current_user.total_study_minutes or 0
    hours_val = round(total_minutes / 60, 1)
    
    # 3. Today's Progress
    today = datetime.utcnow().date()
    plan = StudyPlan.query.filter_by(user_id=user_id, date=today).first()
    daily_progress = 0
    if plan:
        try:
            tasks = json.loads(plan.topics)
            # Handle migration if needed
            tasks = [t if isinstance(t, dict) else {'completed': False} for t in tasks]
            if tasks:
                completed = sum(1 for t in tasks if t.get('completed', False))
                daily_progress = int((completed / len(tasks)) * 100)
        except:
            daily_progress = 0

    # 4. Recent Activity (Merge Notes & Scores)
    recent_activity = []
    
    recent_notes = RevisionNote.query.filter_by(user_id=user_id).order_by(RevisionNote.created_at.desc()).limit(3).all()
    for note in recent_notes:
        recent_activity.append({
            'type': 'note',
            'text': f"Created note: {note.topic}",
            'time': note.created_at.strftime("%H:%M"),
            'date': note.created_at
        })
        
    recent_scores = MCQScore.query.filter_by(user_id=user_id).order_by(MCQScore.created_at.desc()).limit(3).all()
    for score in recent_scores:
        recent_activity.append({
            'type': 'quiz',
            'text': f"Scored {score.score}/{score.total_questions} in {score.topic}",
            'time': score.created_at.strftime("%H:%M"),
            'date': score.created_at
        })
    
    # Sort combined activity by date desc
    recent_activity.sort(key=lambda x: x['date'], reverse=True)
    recent_activity = recent_activity[:5]

    # 5. Weak Topics (Avg Score < 60%)
    topic_performance = {}
    for s in scores:
        if s.topic not in topic_performance:
            topic_performance[s.topic] = {'score': 0, 'total': 0}
        topic_performance[s.topic]['score'] += s.score
        topic_performance[s.topic]['total'] += s.total_questions
    
    weak_topics = []
    for topic, data in topic_performance.items():
        if data['total'] > 0:
            percentage = (data['score'] / data['total']) * 100
            if percentage < 60:
                weak_topics.append({'topic': topic, 'score': int(percentage)})

    # 6. Motivation & AI Rec
    quotes = [
        "Success is sum of small efforts, repeated day in and day out.",
        "Don't watch the clock; do what it does. Keep going.",
        "The secret of getting ahead is getting started.",
        "It always seems impossible until it's done."
    ]
    motivation = random.choice(quotes)
    
    ai_rec = "Review your notes" # Default
    if weak_topics:
        ai_rec = f"Revise {weak_topics[0]['topic']} - Score is low ({weak_topics[0]['score']}%)"
    elif daily_progress < 100:
        ai_rec = "Complete your daily planner tasks"
    else:
        ai_rec = "Great job! Try a coding challenge now."

    # 7. Time Spent (Real-time)
    total_minutes = current_user.total_study_minutes or 0
    t_hours = total_minutes // 60
    t_minutes = total_minutes % 60
    time_spent = f"{t_hours}h {t_minutes}m"

    # 8. Topic-wise Progress Bars
    topic_progress = []
    for topic, data in topic_performance.items():
        if data['total'] > 0:
            pct = int((data['score'] / data['total']) * 100)
            pct = min(pct, 100)
            topic_progress.append({'topic': topic, 'pct': pct})
    topic_progress.sort(key=lambda x: x['pct'])

    return render_template('dashboard.html', 
                           user=current_user, 
                           notes_count=notes_count, 
                           avg_score=avg_score, 
                           problems_solved=problems_solved,
                           streak=streak,
                           daily_progress=daily_progress,
                           recent_activity=recent_activity,
                           weak_topics=weak_topics,
                           motivation=motivation,
                           ai_rec=ai_rec,
                           time_spent=time_spent,
                           topic_progress=topic_progress)

@app.route('/logout')
@login_required
def logout():
    # Invalidate the session token in DB
    current_user.session_token = None
    db.session.commit()
    logout_user()
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('home'))

@app.route('/revision', methods=['GET', 'POST'])
@login_required
def revision():
    if request.method == 'POST':
        topic = request.form.get('topic')
        content = request.form.get('content')
        phase = request.form.get('phase', 'Phase 1')
        tags = request.form.get('tags', '')
        
        note = RevisionNote(
            topic=topic, 
            content=content, 
            phase=phase,
            tags=tags,
            user_id=current_user.id
        )
        db.session.add(note)
        db.session.commit()
        flash('Note created successfully!', 'success')
        return redirect(url_for('revision'))
    
    # Search & Filter
    search_query = request.args.get('q', '')
    query = RevisionNote.query.filter_by(user_id=current_user.id)
    
    if search_query:
        query = query.filter(
            (RevisionNote.topic.contains(search_query)) | 
            (RevisionNote.tags.contains(search_query)) |
            (RevisionNote.content.contains(search_query))
        )
        
    # Sort: Pinned first, then Updated Descending
    notes = query.order_by(RevisionNote.is_pinned.desc(), RevisionNote.updated_at.desc()).all()
    return render_template('revision.html', notes=notes, search_query=search_query)

@app.route('/note/<int:note_id>/delete', methods=['POST'])
@login_required
def delete_note(note_id):
    note = RevisionNote.query.get_or_404(note_id)
    if note.user_id != current_user.id:
        abort(403)
    db.session.delete(note)
    db.session.commit()
    flash('Note deleted.', 'info')
    return redirect(url_for('revision'))

@app.route('/note/<int:note_id>/pin', methods=['POST'])
@login_required
def pin_note(note_id):
    note = RevisionNote.query.get_or_404(note_id)
    if note.user_id != current_user.id:
        abort(403)
    note.is_pinned = not note.is_pinned
    db.session.commit()
    return jsonify({'success': True, 'is_pinned': note.is_pinned})

@app.route('/note/<int:note_id>/edit', methods=['POST'])
@login_required
def edit_note(note_id):
    note = RevisionNote.query.get_or_404(note_id)
    if note.user_id != current_user.id:
        abort(403)
    
    note.topic = request.form.get('topic')
    note.content = request.form.get('content')
    note.phase = request.form.get('phase')
    note.tags = request.form.get('tags')
    
    db.session.commit()
    flash('Note updated.', 'success')
    return redirect(url_for('revision'))



@app.route('/coding')
@login_required
def coding():
    topic = request.args.get('topic')
    challenge = None
    if topic:
        from utils import CodingRecommender
        recommender = CodingRecommender()
        challenge = recommender.generate_challenge(topic, difficulty="medium")
    
    return render_template('coding.html', challenge=challenge)

# ==============================
# AI Tutor — Chat with Persistent History
# ==============================
_ai_tutor = None

def _get_tutor():
    """Return a singleton AITutor (models already cached by ModelCache)."""
    global _ai_tutor
    if _ai_tutor is None:
        from utils import AITutor
        _ai_tutor = AITutor()
    return _ai_tutor

@app.route('/chat')
@login_required
def chat():
    return render_template('chat.html')

@app.route('/chat/conversations', methods=['GET'])
@login_required
def chat_conversations():
    """Return all conversations for the current user."""
    convos = ChatConversation.query.filter_by(user_id=current_user.id)\
        .order_by(ChatConversation.updated_at.desc()).all()
    return jsonify([{
        'id': c.id,
        'title': c.title,
        'updated_at': c.updated_at.strftime('%b %d, %I:%M %p') if c.updated_at else '',
        'message_count': len(c.messages)
    } for c in convos])

@app.route('/chat/conversations', methods=['POST'])
@login_required
def chat_new_conversation():
    """Create a new conversation."""
    convo = ChatConversation(user_id=current_user.id, title='New Chat')
    db.session.add(convo)
    db.session.commit()
    return jsonify({'id': convo.id, 'title': convo.title})

@app.route('/chat/conversations/<int:convo_id>', methods=['GET'])
@login_required
def chat_get_conversation(convo_id):
    """Return all messages in a conversation."""
    convo = ChatConversation.query.filter_by(id=convo_id, user_id=current_user.id).first_or_404()
    return jsonify({
        'id': convo.id,
        'title': convo.title,
        'messages': [{
            'role': m.role,
            'content': m.content,
            'created_at': m.created_at.strftime('%I:%M %p') if m.created_at else ''
        } for m in convo.messages]
    })

@app.route('/chat/conversations/<int:convo_id>', methods=['DELETE'])
@login_required
def chat_delete_conversation(convo_id):
    """Delete a conversation and all its messages."""
    convo = ChatConversation.query.filter_by(id=convo_id, user_id=current_user.id).first_or_404()
    db.session.delete(convo)
    db.session.commit()
    return jsonify({'status': 'deleted'})

@app.route('/chat_api', methods=['POST'])
@login_required
@limiter.limit("30 per minute")
def chat_api():
    data = request.get_json(silent=True) or {}
    user_message = (data.get('message') or '').strip()
    convo_id = data.get('conversation_id')
    model_type = data.get('model_type', 'fast')
    if not user_message:
        return jsonify({"response": "Please type a message!"})

    # Get or create conversation
    convo = None
    if convo_id:
        convo = ChatConversation.query.filter_by(id=convo_id, user_id=current_user.id).first()
    if not convo:
        convo = ChatConversation(user_id=current_user.id, title='New Chat')
        db.session.add(convo)
        db.session.commit()

    # 1. Fetch user context for RAG
    notes    = RevisionNote.query.filter_by(user_id=current_user.id).all()
    codes    = CodeSubmission.query.filter_by(user_id=current_user.id).all()
    roadmaps = StudyRoadmap.query.filter_by(user_id=current_user.id).all()

    user_data = {
        'notes':    [n.content for n in notes],
        'code':     [c.code    for c in codes],
        'syllabus': [r.syllabus_text for r in roadmaps]
    }

    # 2. Build history from DB (last 6 messages = 3 turns)
    recent_msgs = ChatMessage.query.filter_by(conversation_id=convo.id)\
        .order_by(ChatMessage.created_at.desc()).limit(6).all()
    recent_msgs.reverse()
    history = [m.content for m in recent_msgs]

    # 3. Build KB and get response from singleton tutor
    tutor = _get_tutor()
    tutor.build_knowledge_base(user_data)
    api_key = app.config.get('GEMINI_API_KEY')
    bot_response = tutor.get_response(user_message, history, api_key=api_key, model_type=model_type)

    # 4. Save both messages to DB
    db.session.add(ChatMessage(conversation_id=convo.id, role='user', content=user_message))
    db.session.add(ChatMessage(conversation_id=convo.id, role='ai', content=bot_response))

    # 5. Auto-generate title from first user message
    if convo.title == 'New Chat':
        convo.title = user_message[:50] + ('...' if len(user_message) > 50 else '')

    convo.updated_at = datetime.utcnow()
    db.session.commit()

    return jsonify({
        "response": bot_response,
        "conversation_id": convo.id,
        "title": convo.title
    })

@app.route('/analytics')
@login_required
def analytics():
    return render_template('analytics.html')

@app.route('/api/analytics/dashboard')
@login_required
def analytics_dashboard_api():
    user_id = current_user.id
    
    # --- 1. Fetch Data ---
    scores = MCQScore.query.filter_by(user_id=user_id).all()
    notes = RevisionNote.query.filter_by(user_id=user_id).all()
    codes = CodeSubmission.query.filter_by(user_id=user_id).all()
    
    # --- 2. Calculate Metrics ---
    
    # Accuracy
    total_score_val = sum(s.score for s in scores)
    avg_accuracy = int(total_score_val / len(scores)) if scores else 0
    
    # Questions Solved (Quiz Questions + Coding Problems)
    total_questions = sum(s.total_questions for s in scores) + len(codes)
    
    # Study Hours (Heuristic: Note=20m, Quiz=10m, Code=15m)
    hours_val = (len(notes) * 20 + len(scores) * 10 + len(codes) * 15) / 60
    hours_val = round(hours_val, 1)
    
    # --- 3. Topic Performance ---
    topic_stats = {}
    for s in scores:
        if s.topic not in topic_stats:
            topic_stats[s.topic] = {'score_sum': 0, 'count': 0, 'attempts': 0}
        topic_stats[s.topic]['score_sum'] += s.score
        topic_stats[s.topic]['count'] += 1
        topic_stats[s.topic]['attempts'] += 1
        
    topics_list = []
    for topic, data in topic_stats.items():
        acc = int(data['score_sum'] / data['count'])
        color = '#10B981' # Green
        if acc < 40: color = '#EF4444' # Red
        elif acc < 70: color = '#F59E0B' # Yellow
        
        topics_list.append({
            'name': topic,
            'accuracy': acc,
            'color': color,
            'attempts': data['attempts']
        })
    
    # --- 4. Last 7 Days Activity (Chart) ---
    today = datetime.utcnow().date()
    last_7_days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    chart_labels = [d.strftime("%a") for d in last_7_days]
    chart_data = []
    
    for d in last_7_days:
        # Count activity for this day
        day_score = 0
        day_score += sum(0.3 for n in notes if n.created_at.date() == d)
        day_score += sum(0.15 for s in scores if s.created_at.date() == d)
        day_score += sum(0.25 for c in codes if c.timestamp.date() == d)
        chart_data.append(round(day_score, 1))

    # --- 5. Insight ---
    weakest = min(topics_list, key=lambda x: x['accuracy']) if topics_list else None
    insight = {
        'text': "Keep consistency! You're doing great.",
        'action': None
    }
    if weakest and weakest['accuracy'] < 50:
        insight = {
            'text': f"Your performance in {weakest['name']} is a bit low ({weakest['accuracy']}%)",
            'action': f"Revise {weakest['name']}"
        }
    elif avg_accuracy > 80:
        insight = {
            'text': "Excellent accuracy! Time to tackle harder problems.",
            'action': "Try Coding Challenge"
        }

    # Construct Response
    response_data = {
        'metrics': {
            'accuracy': {'value': avg_accuracy, 'change': 5}, # Mock change
            'questions': {'value': total_questions, 'change': 12}, 
            'hours': {'value': hours_val, 'change': 1.5}
        },
        'insight': insight,
        'topics': topics_list,
        'chart': {
            'labels': chart_labels,
            'data': chart_data,
            'goal': 2.0 # Daily goal lines
        }
    }
    
    return jsonify(response_data)


@app.route('/run_code', methods=['POST'])
@login_required
@limiter.limit("20 per minute")
def run_code():
    data = request.json
    code = data.get('code')
    language = data.get('language', 'python')
    import time
    start_time = time.time()

    # Judge0 CE language IDs
    lang_ids = {
        'python': 71,      # Python 3.8.1
        'javascript': 63,  # Node.js 12.14.0
        'java': 62          # Java (OpenJDK 13.0.1)
    }
    lang_id = lang_ids.get(language.lower(), 71)

    # Try Judge0 CE API first
    try:
        # Submit code
        submit_url = "https://judge0-ce.p.sulu.sh/submissions?base64_encoded=false&wait=true"
        payload = {
            "source_code": code,
            "language_id": lang_id,
            "stdin": ""
        }
        response = requests.post(submit_url, json=payload, timeout=15,
                                 headers={"Content-Type": "application/json"})
        result = response.json()

        end_time = time.time()
        duration = round((end_time - start_time) * 1000, 2)

        stdout = result.get('stdout', '') or ''
        stderr = result.get('stderr', '') or ''
        compile_err = result.get('compile_output', '') or ''
        
        if compile_err and not stderr:
            stderr = compile_err

        return jsonify({
            "success": not stderr,
            "stdout": stdout,
            "stderr": stderr,
            "time": f"{duration}ms",
            "memory": f"{result.get('memory', 'N/A')} KB"
        })
    except Exception as api_error:
        # Fallback: Run Python locally (only for Python)
        if language.lower() == 'python':
            try:
                import subprocess
                proc = subprocess.run(
                    ['python', '-c', code],
                    capture_output=True, text=True, timeout=10,
                    cwd=app.root_path
                )
                end_time = time.time()
                duration = round((end_time - start_time) * 1000, 2)
                return jsonify({
                    "success": proc.returncode == 0,
                    "stdout": proc.stdout or '(No output)',
                    "stderr": proc.stderr,
                    "time": f"{duration}ms",
                    "memory": "Local"
                })
            except subprocess.TimeoutExpired:
                return jsonify({"success": False, "stdout": "", "stderr": "Execution timed out (10s limit)", "time": "10000ms"})
            except Exception as e2:
                return jsonify({"success": False, "stdout": "", "stderr": f"Local execution error: {str(e2)}", "time": "0ms"})

        return jsonify({
            "success": False,
            "stdout": "",
            "stderr": f"Code execution API unavailable. Error: {str(api_error)}",
            "time": "0ms"
        })

@app.route('/explain_code', methods=['POST'])
@login_required
@limiter.limit("15 per minute")
def explain_code():
    data = request.json
    code = data.get('code')
    language = data.get('language')
    
    # Mock AI Explanation
    explanation = "This code appears to be a " + language + " script.\n\n"
    if "def " in code or "function" in code:
        explanation += "- It defines one or more functions to encapsulate logic.\n"
    if "print" in code or "console.log" in code:
        explanation += "- It outputs data to the console.\n"
    if "for " in code or "while " in code:
        explanation += "- It uses loops for iteration.\n"
    if "if " in code:
        explanation += "- It contains conditional logic.\n"
        
    explanation += "\nKey Tip: Ensure your syntax matches the " + language + " standard version. Check for missing colons or braces."
    
    return jsonify({"explanation": explanation})

@app.route('/generate_notes', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def generate_notes():
    from utils import TextSummarizer
    
    summarizer = TextSummarizer()
    text = ""
    
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            # Handle PDF
            text = summarizer.extract_text_from_pdf(file)
            if text.startswith("Error"):
                return jsonify({"error": text}), 400
    else:
        # Handle Topic/Text
        data = request.form
        text = data.get('text', '')
        topic = data.get('topic', '')
        
        # If only topic is provided, we can't really "generate" from scratch without an LLM. 
        # But for this feature, we expect text input or we mock a "lookup".
        if not text and topic:
             # For demo, we'll just say we need text.
             # Or we can use the topic as a seed if we had a knowledge base.
             return jsonify({"notes": [f"Please provide text content to summarize for topic: {topic}"]})

    summary = summarizer.summarize(text, api_key=app.config.get('GEMINI_API_KEY'))
    return jsonify({"notes": summary})

@app.route('/mcq', methods=['GET', 'POST'])
@login_required
def mcq():
    from utils import MCQGenerator
    from flask import session as flask_session
    mcq_gen = MCQGenerator()

    DEFAULT_QUESTIONS = [
        {
            'id': 1, 'question': 'What is the full form of SQL?',
            'options': ['Structured Query Language', 'Simple Query Logic', 'System Question List', 'Syntax Query List'],
            'answer': 'Structured Query Language',
            'difficulty': 'Easy', 'badge_color': 'success', 'topic': 'DBMS', 'subtopic': 'Basics',
            'hint': 'It is a standard language for relational databases.',
            'explanation': 'SQL (Structured Query Language) is used to communicate with and manage relational databases.'
        },
        {
            'id': 2, 'question': 'Which data structure follows the LIFO (Last In, First Out) principle?',
            'options': ['Queue', 'Stack', 'Linked List', 'Tree'],
            'answer': 'Stack',
            'difficulty': 'Easy', 'badge_color': 'success', 'topic': 'Data Structures', 'subtopic': 'Concept',
            'hint': 'Think of a stack of plates — you always take from the top.',
            'explanation': 'A Stack follows LIFO — the last element pushed is the first to be popped, just like a stack of plates.'
        },
        {
            'id': 3, 'question': 'In Python, which keyword is used to define a function?',
            'options': ['func', 'define', 'def', 'function'],
            'answer': 'def',
            'difficulty': 'Easy', 'badge_color': 'success', 'topic': 'Python', 'subtopic': 'Syntax',
            'hint': 'It is a 3-letter abbreviation of the word "define".',
            'explanation': 'In Python, the `def` keyword declares a function: `def my_function():`. It stands for "define".'
        },
        {
            'id': 4, 'question': 'What is the output of evaluating 3 * "abc" in Python?',
            'options': ['abcabcabc', '3abc', 'Error', 'abc3'],
            'answer': 'abcabcabc',
            'difficulty': 'Medium', 'badge_color': 'warning', 'topic': 'Python', 'subtopic': 'Operators',
            'hint': 'Python allows multiplying a string by an integer to repeat it.',
            'explanation': 'In Python, multiplying a string by an integer repeats it: `3 * "abc"` gives `"abcabcabc"`.'
        },
        {
            'id': 5, 'question': 'Which sorting algorithm guarantees O(n log n) worst-case time complexity?',
            'options': ['Bubble Sort', 'Quick Sort', 'Merge Sort', 'Insertion Sort'],
            'answer': 'Merge Sort',
            'difficulty': 'Hard', 'badge_color': 'danger', 'topic': 'Algorithms', 'subtopic': 'Sorting',
            'hint': 'This algorithm uses divide-and-conquer and is always stable.',
            'explanation': 'Merge Sort has guaranteed O(n log n) time complexity in all cases by always dividing the array in half.'
        },
    ]

    # ── QUIZ SUBMISSION ──────────────────────────────────────────────────────
    if request.method == 'POST' and request.form.get('action') == 'submit':
        score = 0
        total = 0
        results_details = []

        for key in request.form:
            if key.startswith('q') and key[1:].isdigit():
                q_id = key[1:]
                user_answer = request.form.get(key, '').strip()
                correct_answer = request.form.get(f'ans_{q_id}', '').strip()
                question_text = request.form.get(f'txt_{q_id}', '')

                if correct_answer:
                    total += 1
                    is_correct = (user_answer == correct_answer)
                    if is_correct:
                        score += 1
                    results_details.append({
                        'id': q_id,
                        'question': question_text,
                        'user_answer': user_answer,
                        'correct_answer': correct_answer,
                        'is_correct': is_correct
                    })

        if total == 0:
            # Fallback: score against default questions if hidden fields missing
            for q in DEFAULT_QUESTIONS:
                user_answer = request.form.get(f'q{q["id"]}', '').strip()
                correct_answer = q['answer']
                total += 1
                is_correct = (user_answer == correct_answer)
                if is_correct:
                    score += 1
                results_details.append({
                    'id': str(q['id']),
                    'question': q['question'],
                    'user_answer': user_answer,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct
                })

        percentage = int((score / total) * 100) if total > 0 else 0

        mcq_score = MCQScore(
            topic="Mixed/Generated", score=percentage,
            total_questions=total, user_id=current_user.id
        )
        db.session.add(mcq_score)
        db.session.commit()

        encouragement = "Good effort!"
        if percentage >= 80:
            encouragement = "Great job! Keep it up! \U0001f31f"
        elif percentage < 50:
            encouragement = "Keep practicing! \U0001f4aa"

        # Restore questions from session for review display
        questions_for_review = flask_session.pop('last_mcq_questions', DEFAULT_QUESTIONS)

        return render_template('mcq.html',
            questions=questions_for_review,
            result={
                'score': score, 'total': total,
                'percentage': percentage,
                'encouragement': encouragement,
                'details': results_details
            }
        )

    # ── QUIZ GENERATION ──────────────────────────────────────────────────────
    questions = []
    if request.method == 'POST' and request.form.get('action') == 'generate':
        text_input = request.form.get('source_text', '').strip()
        difficulty = request.form.get('difficulty', 'medium')
        if text_input:
            questions = mcq_gen.generate_mcqs(text_input, difficulty=difficulty, api_key=app.config.get('GEMINI_API_KEY'))

    if not questions:
        questions = DEFAULT_QUESTIONS
        if request.method == 'POST' and request.form.get('action') == 'generate':
            flash('Could not extract enough content to generate questions. Showing default quiz. Try pasting more detailed text.', 'warning')

    # Save generated questions in session so they survive the submit round-trip
    flask_session['last_mcq_questions'] = questions

    return render_template('mcq.html', questions=questions)

import uuid

@app.route('/planner')
@login_required
def planner():
    today = datetime.utcnow().date()
    plan = StudyPlan.query.filter_by(user_id=current_user.id, date=today).first()
    
    tasks = []
    if plan and plan.topics:
        try:
            tasks = json.loads(plan.topics)
        except:
            tasks = []
            
    # Calculate progress for initial render
    total = len(tasks)
    completed = sum(1 for t in tasks if t.get('completed'))
    progress = int((completed / total) * 100) if total > 0 else 0
    
    return render_template('planner.html', tasks=tasks, progress=progress, today=today)

@app.route('/planner/action', methods=['POST'])
@login_required
def planner_action():
    action = request.form.get('action')
    today = datetime.utcnow().date()
    plan = StudyPlan.query.filter_by(user_id=current_user.id, date=today).first()
    
    if not plan:
        plan = StudyPlan(user_id=current_user.id, date=today, topics=json.dumps([]))
        db.session.add(plan)
        db.session.commit()
    
    tasks = json.loads(plan.topics) if plan.topics else []
    
    if action == 'add':
        text = request.form.get('task_text')
        duration = request.form.get('duration', '30m')
        if text:
            new_task = {
                'id': str(uuid.uuid4()),
                'text': text,
                'duration': duration,
                'completed': False,
                'time_completed': None
            }
            tasks.append(new_task)
            flash('Task added successfully.', 'success')
            
    elif action == 'delete':
        task_id = request.form.get('task_id')
        tasks = [t for t in tasks if t.get('id') != task_id]
        flash('Task removed.', 'info')
        
    elif action == 'toggle':
        task_id = request.form.get('task_id')
        for t in tasks:
            if t.get('id') == task_id:
                t['completed'] = not t['completed']
                if t['completed']:
                    # Store clear time like "Completed at 7:30 PM"
                    t['time_completed'] = datetime.utcnow().strftime("%I:%M %p")
                else:
                    t['time_completed'] = None
                break
                
    elif action == 'suggest':
        api_key = app.config.get('GEMINI_API_KEY')
        suggestion = "Review yesterday's notes"
        if api_key and len(api_key) > 10:
            try:
                current_tasks = [t.get('text', '') for t in tasks]
                tasks_str = ", ".join(current_tasks) if current_tasks else "No tasks currently."
                prompt = f"I am a student planning my study day. My current tasks: {tasks_str}. Suggest exactly one short, actionable study task (max 8 words) that I should add to my plan. Don't add quotes or intro."
                
                if api_key.startswith("gsk_"):
                    from groq import Groq
                    client = Groq(api_key=api_key)
                    completion = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    suggestion = completion.choices[0].message.content.strip().replace('"', '')
                else:
                    from gemini_helper import generate_gemini_content
                    response = generate_gemini_content(api_key, prompt, model_type="fast")
                    if response and response.text:
                        suggestion = response.text.strip().replace('"', '')
            except Exception as e:
                print("API Error for suggestions:", e)
                suggestions = ["Revise 5 SQL queries", "Practice 1 Array problem", "Read about HTTP methods", "Review yesterdays notes"]
                import random
                suggestion = random.choice(suggestions)
        else:
            suggestions = ["Revise 5 SQL queries", "Practice 1 Array problem", "Read about HTTP methods", "Review yesterdays notes"]
            import random
            suggestion = random.choice(suggestions)
            
        new_task = {
            'id': str(uuid.uuid4()),
            'text': f"AI Suggestion: {suggestion}",
            'duration': '30m',
            'completed': False,
            'time_completed': None
        }
        tasks.append(new_task)
        flash('AI Task added!', 'success')

    plan.topics = json.dumps(tasks)
    db.session.commit()
    
    return redirect(url_for('planner'))

@app.route('/study_plan', methods=['GET', 'POST'])
@login_required
def study_plan():
    roadmap = StudyRoadmap.query.filter_by(user_id=current_user.id).first()
    plan = []
    
    if roadmap:
         try:
             plan = json.loads(roadmap.roadmap_json)
         except:
             plan = []

    if request.method == 'POST':
        text = request.form.get('syllabus_text')
        if text:
            from utils import AnalyticsEngine
            engine = AnalyticsEngine()
            plan = engine.generate_study_plan_from_text(text)
            
            if not roadmap:
                roadmap = StudyRoadmap(user_id=current_user.id, syllabus_text=text, roadmap_json=json.dumps(plan))
                db.session.add(roadmap)
            else:
                roadmap.syllabus_text = text
                roadmap.roadmap_json = json.dumps(plan)
            
            db.session.commit()
            flash('Your study plan is ready!', 'success')
            
    return render_template('study_plan.html', plan=plan)

@app.route('/coding/suggest', methods=['POST'])
@login_required
def coding_suggest():
    from utils import CodingRecommender
    recommender = CodingRecommender()
    
    data = request.json
    topic = data.get('topic', '')
    
    suggestions = recommender.suggest_problems(topic)
    return jsonify(suggestions)

@app.route('/save_generated_notes', methods=['POST'])
@login_required
def save_generated_notes():
    data = request.json
    topic = data.get('topic', 'Generated Note')
    content = "\n".join([f"- {point}" for point in data.get('notes', [])])
    
    note = RevisionNote(topic=topic, content=content, user_id=current_user.id)
    db.session.add(note)
    db.session.commit()
    
    return jsonify({"success": True})

# ==========================================
# PDF Intelligence Routes
# ==========================================

# In-memory store for current PDF session per user
_pdf_sessions = {}

@app.route('/pdf_analyzer')
@login_required
def pdf_analyzer():
    # Get user's previously uploaded PDFs
    pdfs = PDFDocument.query.filter_by(user_id=current_user.id).order_by(PDFDocument.created_at.desc()).limit(10).all()
    return render_template('pdf_analyzer.html', pdfs=pdfs)

@app.route('/api/pdf/upload', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def pdf_upload():
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['pdf_file']
    if file.filename == '' or not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Please upload a valid PDF file"}), 400

    try:
        # Reset stream position — critical! Flask may have already read headers
        file.stream.seek(0)

        from pdf_intelligence import PDFIntelligence
        engine = PDFIntelligence()
        result = engine.process_pdf(file.stream)

        if "error" in result:
            print(f"[PDF Upload] Extraction error: {result['error']}")
            return jsonify({"error": result["error"]}), 400

        if not result.get("text") or len(result["text"].strip()) < 10:
            return jsonify({"error": "Could not extract meaningful text from this PDF. It may be a scanned image PDF."}), 400

        # Save to database
        pdf_doc = PDFDocument(
            user_id=current_user.id,
            filename=file.filename,
            extracted_text=result["text"],
            num_pages=result["pages"],
            word_count=result["word_count"]
        )
        db.session.add(pdf_doc)
        db.session.commit()

        # Store in session for quick access
        _pdf_sessions[current_user.id] = result["text"]

        return jsonify({
            "success": True,
            "pdf_id": pdf_doc.id,
            "filename": file.filename,
            "pages": result["pages"],
            "word_count": result["word_count"]
        })
    except Exception as e:
        print(f"[PDF Upload] EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500

@app.route('/api/pdf/load/<int:pdf_id>', methods=['POST'])
@login_required
def pdf_load(pdf_id):
    pdf_doc = PDFDocument.query.get_or_404(pdf_id)
    if pdf_doc.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    _pdf_sessions[current_user.id] = pdf_doc.extracted_text
    return jsonify({"success": True, "filename": pdf_doc.filename, "pages": pdf_doc.num_pages, "word_count": pdf_doc.word_count})

@app.route('/api/pdf/summarize', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def pdf_summarize():
    text = _pdf_sessions.get(current_user.id)
    if not text:
        return jsonify({"error": "No PDF loaded. Upload a PDF first."}), 400

    from pdf_intelligence import PDFIntelligence
    engine = PDFIntelligence()
    summary = engine.get_summary(text)
    return jsonify({"summary": summary})

@app.route('/api/pdf/keypoints', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def pdf_keypoints():
    text = _pdf_sessions.get(current_user.id)
    if not text:
        return jsonify({"error": "No PDF loaded. Upload a PDF first."}), 400

    from pdf_intelligence import PDFIntelligence
    engine = PDFIntelligence()
    points = engine.get_key_points(text)
    return jsonify({"keypoints": points})

@app.route('/api/pdf/notes', methods=['POST'])
@login_required
def pdf_notes():
    text = _pdf_sessions.get(current_user.id)
    if not text:
        return jsonify({"error": "No PDF loaded. Upload a PDF first."}), 400

    from pdf_intelligence import PDFIntelligence
    engine = PDFIntelligence()
    notes = engine.get_short_notes(text)
    return jsonify({"notes": notes})

@app.route('/api/pdf/mcqs', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def pdf_mcqs():
    text = _pdf_sessions.get(current_user.id)
    if not text:
        return jsonify({"error": "No PDF loaded. Upload a PDF first."}), 400

    data = request.json or {}
    difficulty = data.get('difficulty', 'medium')
    num_q = min(int(data.get('num_questions', 10)), 20)

    from pdf_intelligence import PDFIntelligence
    engine = PDFIntelligence()
    mcqs = engine.get_mcqs(text, num_q, difficulty)
    return jsonify({"mcqs": mcqs})

@app.route('/api/pdf/ask', methods=['POST'])
@login_required
@limiter.limit("15 per minute")
def pdf_ask():
    text = _pdf_sessions.get(current_user.id)
    if not text:
        return jsonify({"error": "No PDF loaded. Upload a PDF first."}), 400

    data = request.json or {}
    question = data.get('question', '')
    if not question:
        return jsonify({"error": "Please provide a question."}), 400

    api_key = app.config.get('GEMINI_API_KEY')
    from pdf_intelligence import PDFIntelligence
    engine = PDFIntelligence()
    answer = engine.ask_question(question, text, api_key=api_key)
    return jsonify({"answer": answer})

@app.route('/api/pdf/delete/<int:pdf_id>', methods=['POST'])
@login_required
def pdf_delete(pdf_id):
    pdf_doc = PDFDocument.query.get_or_404(pdf_id)
    if pdf_doc.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    db.session.delete(pdf_doc)
    db.session.commit()
    return jsonify({"success": True})

# ==========================================
# Study Groups - Real-Time Collaboration
# ==========================================

def generate_invite_code():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

@app.route('/study_groups')
@login_required
def study_groups():
    memberships = GroupMember.query.filter_by(user_id=current_user.id).all()
    my_groups = [m.group for m in memberships if m.group.is_active]
    return render_template('study_groups.html', groups=my_groups)

@app.route('/api/groups/create', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def create_group():
    data = request.json
    name = data.get('name', '').strip()
    desc = data.get('description', '')
    if not name:
        return jsonify({"error": "Group name required"}), 400

    code = generate_invite_code()
    while StudyGroup.query.filter_by(invite_code=code).first():
        code = generate_invite_code()

    group = StudyGroup(name=name, description=desc, invite_code=code, created_by=current_user.id)
    db.session.add(group)
    db.session.flush()
    member = GroupMember(group_id=group.id, user_id=current_user.id, role='admin')
    db.session.add(member)
    sys_msg = GroupMessage(group_id=group.id, user_id=current_user.id,
                           message=f"{current_user.username} created the group", msg_type='system')
    db.session.add(sys_msg)
    db.session.commit()
    return jsonify({"success": True, "group_id": group.id, "invite_code": code})

@app.route('/api/groups/join', methods=['POST'])
@login_required
def join_group():
    data = request.json
    code = data.get('code', '').strip().upper()
    group = StudyGroup.query.filter_by(invite_code=code, is_active=True).first()
    if not group:
        return jsonify({"error": "Invalid invite code"}), 404
    existing = GroupMember.query.filter_by(group_id=group.id, user_id=current_user.id).first()
    if existing:
        return jsonify({"error": "Already a member"}), 400
    member = GroupMember(group_id=group.id, user_id=current_user.id, role='member')
    db.session.add(member)
    sys_msg = GroupMessage(group_id=group.id, user_id=current_user.id,
                           message=f"{current_user.username} joined the group", msg_type='system')
    db.session.add(sys_msg)
    db.session.commit()
    socketio.emit('user_joined', {'username': current_user.username, 'group_id': group.id}, room=f'group_{group.id}')
    return jsonify({"success": True, "group_id": group.id, "group_name": group.name})

@app.route('/group/<int:group_id>')
@login_required
def group_room(group_id):
    group = StudyGroup.query.get_or_404(group_id)
    membership = GroupMember.query.filter_by(group_id=group_id, user_id=current_user.id).first()
    if not membership:
        flash('You are not a member of this group.', 'danger')
        return redirect(url_for('study_groups'))
    members = GroupMember.query.filter_by(group_id=group_id).all()
    messages = GroupMessage.query.filter_by(group_id=group_id).order_by(GroupMessage.created_at.asc()).limit(100).all()
    shared_pdf = PDFDocument.query.get(group.shared_pdf_id) if group.shared_pdf_id else None
    user_pdfs = PDFDocument.query.filter_by(user_id=current_user.id).all()
    return render_template('group_room.html', group=group, members=members,
                           messages=messages, shared_pdf=shared_pdf, user_pdfs=user_pdfs,
                           is_admin=(membership.role == 'admin'), timedelta=timedelta)

@app.route('/api/groups/<int:group_id>/share_pdf', methods=['POST'])
@login_required
def share_pdf_to_group(group_id):
    group = StudyGroup.query.get_or_404(group_id)
    data = request.json
    pdf_id = data.get('pdf_id')
    pdf = PDFDocument.query.get_or_404(pdf_id)
    group.shared_pdf_id = pdf_id
    sys_msg = GroupMessage(group_id=group_id, user_id=current_user.id,
                           message=f"{current_user.username} shared PDF: {pdf.filename}", msg_type='system')
    db.session.add(sys_msg)
    db.session.commit()
    _pdf_sessions[current_user.id] = pdf.extracted_text
    socketio.emit('pdf_shared', {'filename': pdf.filename, 'username': current_user.username}, room=f'group_{group_id}')
    return jsonify({"success": True})

@app.route('/api/groups/<int:group_id>/quiz', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def group_quiz(group_id):
    group = StudyGroup.query.get_or_404(group_id)
    if not group.shared_pdf_id:
        return jsonify({"error": "No shared PDF. Share a PDF first."}), 400
    pdf = PDFDocument.query.get(group.shared_pdf_id)
    if not pdf:
        return jsonify({"error": "Shared PDF not found."}), 404
    data = request.json or {}
    num_q = min(int(data.get('num_questions', 5)), 10)
    from pdf_intelligence import PDFIntelligence
    engine = PDFIntelligence()
    mcqs = engine.get_mcqs(pdf.extracted_text, num_q, 'medium')
    return jsonify({"mcqs": mcqs})

@app.route('/api/groups/<int:group_id>/leave', methods=['POST'])
@login_required
def leave_group(group_id):
    membership = GroupMember.query.filter_by(group_id=group_id, user_id=current_user.id).first()
    if membership:
        db.session.delete(membership)
        sys_msg = GroupMessage(group_id=group_id, user_id=current_user.id,
                               message=f"{current_user.username} left the group", msg_type='system')
        db.session.add(sys_msg)
        db.session.commit()
        socketio.emit('user_left', {'username': current_user.username}, room=f'group_{group_id}')
    return jsonify({"success": True})

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    if current_user.is_authenticated:
        join_room(f"user_{current_user.id}")

@socketio.on('join_group')
def handle_join_group(data):
    room = f"group_{data['group_id']}"
    join_room(room)
    emit('status', {'msg': f"{data.get('username', 'Someone')} is online"}, room=room)

@socketio.on('leave_group')
def handle_leave_group(data):
    room = f"group_{data['group_id']}"
    leave_room(room)

@socketio.on('send_message')
def handle_message(data):
    group_id = data['group_id']
    user_id = data['user_id']
    message = data['message']
    username = data.get('username', 'Unknown')
    msg_type = data.get('type', 'text')

    with app.app_context():
        msg = GroupMessage(group_id=group_id, user_id=user_id, message=message, msg_type=msg_type)
        db.session.add(msg)
        db.session.commit()

        # Emit to group room
        emit('new_message', {
            'username': username,
            'message': message,
            'type': msg_type,
            'time': (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime('%H:%M')
        }, room=f'group_{group_id}')

        # Broadcast global notification to other members
        group = StudyGroup.query.get(group_id)
        if group:
            members = GroupMember.query.filter_by(group_id=group_id).all()
            for member in members:
                if str(member.user_id) != str(user_id):
                    emit('global_notification', {
                        'title': f'New message in {group.name}',
                        'message': f"{username}: {message}",
                        'link': url_for('group_room', group_id=group_id)
                    }, room=f'user_{member.user_id}')

@socketio.on('quiz_score')
def handle_quiz_score(data):
    emit('score_update', {
        'username': data.get('username'),
        'score': data.get('score'),
        'total': data.get('total')
    }, room=f"group_{data['group_id']}")

@socketio.on('annotation')
def handle_annotation(data):
    group_id = data['group_id']
    with app.app_context():
        msg = GroupMessage(group_id=group_id, user_id=data['user_id'],
                           message=data['text'], msg_type='annotation')
        db.session.add(msg)
        db.session.commit()
    emit('new_annotation', {
        'username': data.get('username'),
        'text': data['text'],
        'time': datetime.utcnow().strftime('%H:%M')
    }, room=f'group_{group_id}')

# ==========================================
# Profile & Settings
# ==========================================

ALLOWED_PICTURE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads', 'profiles')

def allowed_picture(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_PICTURE_EXTENSIONS

@app.route('/profile')
@login_required
def profile():
    """Profile & Settings page."""
    return render_template('profile.html')

@app.route('/profile/update', methods=['POST'])
@login_required
def profile_update():
    """Update display name, bio, study goal."""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    username = data.get('username', '').strip()
    bio = data.get('bio', '').strip()
    daily_goal = data.get('daily_study_goal')

    if username:
        if len(username) < 2 or len(username) > 50:
            return jsonify({'success': False, 'error': 'Username must be 2-50 characters'}), 400
        current_user.username = username

    if bio is not None:
        if len(bio) > 300:
            return jsonify({'success': False, 'error': 'Bio must be under 300 characters'}), 400
        current_user.bio = bio

    if daily_goal is not None:
        try:
            daily_goal = int(daily_goal)
            if daily_goal < 10 or daily_goal > 480:
                return jsonify({'success': False, 'error': 'Study goal must be 10-480 minutes'}), 400
            current_user.daily_study_goal = daily_goal
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Invalid study goal value'}), 400

    db.session.commit()
    return jsonify({'success': True, 'message': 'Profile updated successfully'})

@app.route('/profile/change-password', methods=['POST'])
@login_required
def profile_change_password():
    """Change password — requires old password verification."""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    old_password = data.get('old_password', '')
    new_password = data.get('new_password', '')
    confirm_password = data.get('confirm_password', '')

    if not old_password or not new_password:
        return jsonify({'success': False, 'error': 'All fields are required'}), 400

    if new_password != confirm_password:
        return jsonify({'success': False, 'error': 'New passwords do not match'}), 400

    if len(new_password) < 6:
        return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400

    # Verify old password
    if not bcrypt.check_password_hash(current_user.password, old_password):
        return jsonify({'success': False, 'error': 'Current password is incorrect'}), 400

    current_user.password = bcrypt.generate_password_hash(new_password).decode('utf-8')
    db.session.commit()
    return jsonify({'success': True, 'message': 'Password changed successfully'})

@app.route('/profile/upload-picture', methods=['POST'])
@login_required
def profile_upload_picture():
    """Upload profile picture."""
    if 'picture' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['picture']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    if not allowed_picture(file.filename):
        return jsonify({'success': False, 'error': 'Only PNG, JPG, GIF, WEBP files are allowed'}), 400

    # Generate unique filename
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"

    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Delete old picture if exists
    if current_user.profile_picture:
        old_path = os.path.join(UPLOAD_FOLDER, current_user.profile_picture)
        if os.path.exists(old_path):
            os.remove(old_path)

    # Save new picture
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    current_user.profile_picture = filename
    db.session.commit()

    return jsonify({
        'success': True,
        'message': 'Profile picture updated',
        'picture_url': url_for('static', filename=f'uploads/profiles/{filename}')
    })

# ==========================================
# Admin Panel & User Management
# ==========================================

@app.route('/admin/auth', methods=['GET', 'POST'])
@login_required
def admin_auth():
    """Admin re-authentication gate — requires password before admin access."""
    if not current_user.is_admin:
        abort(403)
    if request.method == 'POST':
        password = request.form.get('password', '')
        if bcrypt.check_password_hash(current_user.password, password):
            session['admin_authed_at'] = datetime.utcnow().isoformat()
            flash('Admin access verified.', 'success')
            next_url = session.pop('admin_next', url_for('admin_panel'))
            return redirect(next_url)
        else:
            flash('Incorrect password. Please try again.', 'danger')
    return render_template('admin_auth.html')

@app.route('/admin')
@admin_required
def admin_panel():
    users = User.query.order_by(User.created_at.desc()).all()
    audit_logs = AdminAuditLog.query.order_by(AdminAuditLog.created_at.desc()).limit(20).all()
    stats = {
        'total_users': len(users),
        'students': sum(1 for u in users if u.role == 'student'),
        'admins': sum(1 for u in users if u.role == 'admin'),
        'total_notes': RevisionNote.query.count(),
        'total_quizzes': MCQScore.query.count(),
        'total_pdfs': PDFDocument.query.count(),
        'total_groups': StudyGroup.query.count(),
    }
    return render_template('admin_panel.html', users=users, stats=stats, audit_logs=audit_logs)

@app.route('/admin/user/<int:user_id>/role', methods=['POST'])
@admin_required
def change_user_role(user_id):
    user = User.query.get_or_404(user_id)
    data = request.json
    new_role = data.get('role')
    if new_role not in ('student', 'admin'):
        return jsonify({"error": "Invalid role"}), 400
    if user.id == current_user.id and new_role != 'admin':
        return jsonify({"error": "Cannot demote yourself"}), 400
    old_role = user.role
    user.role = new_role
    _log_admin_action('role_change', target_user_id=user_id, details=f'{old_role} → {new_role}')
    db.session.commit()
    return jsonify({"success": True, "new_role": new_role})

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        return jsonify({"error": "Cannot delete yourself"}), 400
    _log_admin_action('delete_user', target_user_id=user_id, details=f'Disabled user: {user.username} ({user.email})')
    # Soft delete
    user.is_active = False
    user.session_token = None # Force logout
    db.session.commit()
    return jsonify({"success": True})

@app.route('/admin/audit/<int:log_id>/revoke', methods=['POST'])
@admin_required
def revoke_admin_action(log_id):
    log = AdminAuditLog.query.get_or_404(log_id)
    if log.is_revoked:
        return jsonify({"error": "Action already revoked"}), 400
        
    if log.action == 'delete_user':
        user = User.query.get(log.target_user_id)
        if user:
            user.is_active = True
            log.is_revoked = True
            db.session.commit()
            return jsonify({"success": True, "message": "User account restored."})
        return jsonify({"error": "User not found"}), 404
        
    elif log.action == 'role_change':
        user = User.query.get(log.target_user_id)
        if user:
            parts = log.details.split(' → ')
            if len(parts) == 2:
                old_role = parts[0].strip()
                user.role = old_role
                log.is_revoked = True
                db.session.commit()
                return jsonify({"success": True, "message": f"Role reverted to {old_role}."})
        return jsonify({"error": "User or previous role not found"}), 404
        
    return jsonify({"error": "This action cannot be revoked"}), 400

@app.route('/admin/user/<int:user_id>/reset', methods=['POST'])
@admin_required
def reset_user_password(user_id):
    user = User.query.get_or_404(user_id)
    new_pw = bcrypt.generate_password_hash('password123').decode('utf-8')
    user.password = new_pw
    # Invalidate their session too
    user.session_token = None
    _log_admin_action('reset_password', target_user_id=user_id, details=f'Password reset for: {user.username}')
    db.session.commit()
    return jsonify({"success": True, "message": "Password reset to 'password123'"})

# ==========================================
# Error Handlers
# ==========================================

@app.errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403

@app.errorhandler(429)
def rate_limited(e):
    return jsonify({"error": "Too many requests. Please slow down.", "retry_after": str(e.description)}), 429

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Create default admin if no admin exists
        admin = User.query.filter_by(role='admin').first()
        if not admin:
            existing = User.query.filter_by(email='admin@smartrevise.com').first()
            if existing:
                existing.role = 'admin'
            else:
                hashed = bcrypt.generate_password_hash('admin123').decode('utf-8')
                admin = User(username='Admin', email='admin@smartrevise.com', password=hashed, role='admin')
                db.session.add(admin)
            db.session.commit()
            print("[OK] Default admin created: admin@smartrevise.com / admin123")

        # ---- Model Caching: Pre-load all ML models at startup ----
        from model_cache import cache as model_cache
        model_cache.preload_all()

    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
