# 🎓 SmartRevise AI — AI-Powered Intelligent Study Workspace

SmartRevise AI is a state-of-the-art, feature-rich web platform designed to revolutionize exam preparation and structured learning. By combining advanced traditional Machine Learning (NLP, Clustering, Graph algorithms) with Google’s Gemini AI, SmartRevise AI offers students an interactive, personalized workspace containing AI-guided tutors, PDF intelligence engines, coding playgrounds, and real-time collaboration study groups.

---

## 🌟 Key Features

### 1. 📊 Interactive Student Dashboard
*   **Study Analytics at a Glance:** Track total study time, current consecutive day streak, problem-solving counts, and overall MCQ/quiz accuracy.
*   **Daily Goal Tracker:** Set study targets and track daily progress via interactive visual completion meters.
*   **AI Recommendations:** Dynamically suggests weak topics to revise or coding tasks to attempt based on performance metrics.
*   **Recent Activity Feed:** Unified history showing recent revision notes created and quiz scores.

### 2. 📄 PDF Intelligence Engine
Powered by a hybrid algorithm combining traditional NLP with Gemini AI:
*   **Clean Extraction:** Extract text content from multi-page PDFs with line/hyphen correction.
*   **Hybrid Summarizer:** True PageRank-based TextRank summarization to build sentence graphs and identify central points, with Gemini abstractive bullet-point summarization fallback.
*   **Key Points & Named Entity Recognition (NER):** Hybrid keyword/concept extraction using SpaCy's NER and TF-IDF to highlight terms, organizations, laws, and events.
*   **Topic Clustering:** Auto-groups text into semantic study topics using TF-IDF vectorization and KMeans clustering.
*   **Instant Quiz Generation:** Generates multiple-choice and short-answer questions. Includes semantic distractors using WordNet hyponyms and char-ngram cosine similarity.
*   **PDF RAG QA:** Chat directly with your PDF documents via a TF-IDF Retrieval-Augmented Generation (RAG) vector index.

### 3. 🤖 AI Tutor Chat (RAG-Enabled)
*   **Context-Aware Tutor:** RAG pipeline that reads student revision notes, study roadmaps, and coding submissions to provide tailored instruction.
*   **Model Selection:** Choose between **Fast** (highly efficient responses) and **Advanced** (deep conceptual explanations) models.
*   **Persistent Chat History:** Seamlessly create, title, delete, and switch between multi-turn study conversation histories.

### 4. 📝 Active Revision Board & Study Planner
*   **Study Board:** Create, edit, tag, search, and pin study cards and revision notes.
*   **Daily Planner:** Structured study plans auto-scheduled based on dynamic syllabus roadmap generation.

### 5. 💻 Smart Coding Arena
*   **AI Recommendations:** Dynamic programming challenge recommendations (Easy, Medium, Hard, and Algorithmic levels) targeting student weaknesses.
*   **Code Playground:** Interactive editor supporting multiple programming languages with code validation and execution scoring.

### 6. 📈 Advanced Analytics & Insights
*   **Accuracy Analytics:** Visual metrics on average topic accuracies and question completions.
*   **7-Day Study Heatmap:** Chart indicating study consistency over the last week.
*   **Insight Generator:** Personalized feedback pinpointing your weakest topics and recommending direct actions.

### 7. 👥 Real-Time Collaboration Rooms
*   **Study Groups:** Create or join secure study groups using dynamic invite codes.
*   **Socket.IO Chat Rooms:** Real-time messaging, study session coordination, and live annotations/pdf discussions.

### 8. 🛡️ Enterprise-Grade Security & Administration
*   **Advanced Authentication:** Standard secure credentials or Google OAuth sign-in.
*   **Security Hardening:** CSRF token verification, session fingerprint checks (User-Agent + IP) to prevent hijacking, automatic idle timeout, and lockout after 5 consecutive failed login attempts.
*   **Admin Panel:** Comprehensive system dashboard to manage active/inactive users, adjust roles, configure system API keys, and review audit logs.

---

## 🛠️ Technology Stack

*   **Backend Core:** Python, Flask, Flask-SQLAlchemy (ORM), Flask-Migrate (Alembic), Flask-Login, Flask-Bcrypt, Flask-WTF (CSRF), Flask-Limiter, Flask-SocketIO
*   **Frontend UI:** Vanilla HTML5, CSS3 Custom Properties (sleek dark modes, glassmorphism, fluid animations), and Javascript (ES6)
*   **AI & ML Engines:** 
    *   **Generative AI:** Google Gemini API (`google-generativeai`)
    *   **Traditional NLP:** SpaCy (`en_core_web_sm`), NLTK, Scikit-Learn, NetworkX, NumPy
*   **Database:** SQLite (local development) / PostgreSQL (production)
*   **Production Server:** Gunicorn, Eventlet/Gevent (for WebSocket handling)

---

## 📦 Installation & Local Setup

Follow these steps to set up and run SmartRevise AI on your local machine:

### 1. Clone & Navigate
```bash
git clone <repository-url>
cd SmartReviseAi
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLP Models & Resources
SmartRevise AI uses specific NLP resources for text analysis and offline heuristics:
```bash
# Download SpaCy small English model
python -m spacy download en_core_web_sm

# Download NLTK datasets
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### 5. Configure Environment Variables
Create a file named `.env` in the root directory (or copy `.env.example`):
```bash
SECRET_KEY=your_secure_random_hex_string_here
GEMINI_API_KEY=AIzaSy... # Your Google Gemini API Key

# Optional: Uncomment if deploying to a production Postgres server
# DATABASE_URL=postgresql://user:password@localhost:5432/smartrevise
```

### 6. Initialize the Database
Run the startup script to create database tables and seed the default admin account:
```bash
python init_db.py
```
> ⚠️ **Default Admin Credentials:**
> *   **Email:** `admin@smartrevise.com`
> *   **Password:** `admin123`
> *(Please change this immediately in the settings page upon your first login.)*

### 7. Run the Application
Start the Flask local development server:
```bash
python app.py
```
The app will be running locally at `http://127.0.0.1:5000/`.

---

## 🧪 Verification & Testing
To verify that the offline NLP engine, MCQ generator, and study challenge recommendation pipelines are working properly, run the test script:
```bash
python verify_ml.py
```

---

## 🚀 Cloud Deployment (Render)

SmartRevise AI is ready to deploy to Render. The repository includes `render.yaml` and `build.sh` for auto-provisioning:

1.  Create a new **Blueprint** service on Render.
2.  Connect your repository.
3.  Render will auto-detect the service settings, provisioning a web service and a PostgreSQL database.
4.  Configure the environment variables in your Render Dashboard:
    *   `SECRET_KEY` (Auto-generated by Render)
    *   `PYTHON_VERSION` (Set to `3.11.0`)
    *   `GEMINI_API_KEY` (Your Google Gemini Key)
5.  Deploy! The `build.sh` script automatically installs requirements, downloads spaCy & NLTK datasets, and runs migrations.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
