"""
utils.py — ML/NLP Utility Classes for SmartRevise AI
All heavy models are obtained from model_cache.cache (pre-loaded at startup).
"""

import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

from model_cache import cache as model_cache

# ==========================================
# ML/NLP Pipeline Components
# ==========================================

class TextPreprocessor:
    """Step 2: Text Preprocessing"""
    def __init__(self):
        self.stop_words = set([
            "the", "is", "at", "which", "on", "and", "a", "an", "in", "to", "of", "for", "with", "as", 
            "by", "that", "this", "it", "or", "are", "be", "from", "was", "were", "but", "not", "have", "has",
            "can", "will", "if", "than", "then", "we", "you", "they", "he", "she"
        ])

    def clean_text(self, text):
        """Convert to lowercase, remove special characters."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s.]', '', text) # Keep dots for sentence splitting logic check
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize_words(self, text):
        """Tokenize text into words."""
        # Simple split by whitespace after cleaning
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        return clean.split()

    def remove_stopwords(self, words):
        """Remove stop words."""
        return [w for w in words if w not in self.stop_words]

    def get_sentences(self, text):
        """Tokenize text into sentences."""
        return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

class FrequencyAnalyzer:
    """Step 3: Keyword Frequency Analysis"""
    def get_frequent_keywords(self, tokens, top_n=10):
        """Count how often important words appear."""
        counts = Counter(tokens)
        return [word for word, count in counts.most_common(top_n)]

class PatternRecognizer:
    """Step 4: Pattern Recognition"""
    def identify_patterns(self, text):
        """Identify headings, definitions, and concepts."""
        patterns = {
            'definitions': [],
            'headings': [],
            'concepts': []
        }
        
        # Simple heuristics
        sentences = re.split(r'[.!?]\s+', text)
        for sent in sentences:
            # Definition: "X is a Y" or "X refers to Y"
            if " is a " in sent or " refers to " in sent or " defined as " in sent:
                patterns['definitions'].append(sent)
            
            # Concept: Starts with a capitalized word (heuristic, weak in lowercase text but useful in mixed)
            # Since we cleaned text to lowercase in preprocess, we might check original text if available.
            # Here we assume logic on processed sentence.
            pass 
            
        return patterns

# ==========================================
# Feature Classes
# ==========================================

class TextSummarizer:
    """ML-based extractive text summarizer using TF-IDF sentence scoring."""
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.analyzer = FrequencyAnalyzer()
        self.pattern_recognizer = PatternRecognizer()

    def extract_text_from_pdf(self, file_stream):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(file_stream)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            return "Error: PyPDF2 not installed."
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"

    def summarize(self, text, num_sentences=5, api_key=None):
        """Advanced TextRank summarization (PageRank on sentence similarity graph)."""
        if not text:
            return ["No text provided."]
        try:
            from pdf_intelligence import TextRankSummarizer
            return TextRankSummarizer().summarize(text, max_points=num_sentences, api_key=api_key)
        except Exception:
            return self._statistical_summarize(text, num_sentences)

    def _statistical_summarize(self, text, num_sentences=5):
        """Fallback: TF-IDF keyword-scored extractive summarization."""
        sentences = self.preprocessor.get_sentences(text)
        cleaned_text = self.preprocessor.clean_text(text)
        tokens = self.preprocessor.tokenize_words(cleaned_text)
        filtered_tokens = self.preprocessor.remove_stopwords(tokens)
        top_keywords = self.analyzer.get_frequent_keywords(filtered_tokens, top_n=20)
        keyword_set = set(top_keywords)
        sentence_scores = {}
        for sent in sentences:
            score = sum(1 for w in self.preprocessor.tokenize_words(sent) if w in keyword_set)
            if " is a " in sent or " defined " in sent:
                score += 2
            sentence_scores[sent] = score
        ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
        return ranked_sentences[:num_sentences]

class MCQGenerator:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        # Use pre-loaded spaCy from cache
        self.nlp = model_cache.get_spacy()
        self.has_ml = self.nlp is not None

        # Fallback to TFIDF if no ML
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # 1. MCQ Templates
        self.templates = [
            {"template": "What is the primary function or definition of <KEYWORD>?", "type": "Definition"},
            {"template": "Which of the following best describes <KEYWORD>?", "type": "Concept"},
            {"template": "In this context, what does <KEYWORD> refer to?", "type": "Reference"}
        ]

    def _get_distractors(self, word):
        distractors = set()
        try:
            from nltk.corpus import wordnet
            synsets = wordnet.synsets(word)
            if synsets:
                hypernyms = synsets[0].hypernyms()
                if hypernyms:
                    for hypo in hypernyms[0].hyponyms():
                        name = hypo.lemmas()[0].name().replace('_', ' ')
                        if name.lower() != word.lower():
                            distractors.add(name)
        except Exception:
            pass
        return list(distractors)

    def generate_mcqs(self, text, num_questions=5, difficulty="medium", api_key=None):
        if not text: return []
        
        from pdf_intelligence import PDFMCQGenerator
        if api_key:
            questions = PDFMCQGenerator().generate(text, num_questions, difficulty, api_key)
            if questions:
                return questions

        sentences = self.preprocessor.get_sentences(text)
        questions = []
        
        # Determine keywords using SpaCy NER & Noun Chunks
        keywords = []
        if self.has_ml:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"]:
                    keywords.append((ent.text, ent.sent.text))
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) < 3: # Keep it short
                    keywords.append((chunk.text, chunk.sent.text))
        else:
            # Fallback
            words = text.split()
            keywords = [(w, sent) for sent in sentences for w in sent.split() if len(w) > 5]
            
        # Deduplicate keywords
        unique_keywords = {}
        for k, s in keywords:
            if k.lower() not in unique_keywords:
                unique_keywords[k.lower()] = (k, s)
                
        kw_list = list(unique_keywords.values())
        random.shuffle(kw_list)

        for keyword, sent in kw_list:
            if len(questions) >= num_questions: break
            
            template_obj = random.choice(self.templates)
            template = template_obj['template']

            current_diff_label = "Medium"
            badge_color = "warning"
            if difficulty == "easy":
                current_diff_label = "Easy"
                badge_color = "success"
            elif difficulty == "hard":
                current_diff_label = "Hard"
                badge_color = "danger"

            question_text = template.replace("<KEYWORD>", keyword)
            
            # Generate ML distractors
            distractors = self._get_distractors(keyword)
            if len(distractors) < 3:
                # Fallback distractors
                fallback = [k[0] for k in kw_list if k[0].lower() != keyword.lower()]
                random.shuffle(fallback)
                distractors.extend(fallback[:3])
                
            if len(distractors) < 3:
                distractors.extend(["Option A", "Option B", "Option C"])
                
            options = [keyword] + distractors[:3]
            random.shuffle(options)
            
            hint = f"Focus on the context: '{sent[:50]}...'"
            explanation = f"Correct Answer: {keyword}. Based on: '{sent.strip()}'"
            
            questions.append({
                'id': len(questions)+1,
                'question': question_text,
                'options': options,
                'answer': keyword,
                'difficulty': current_diff_label, 
                'badge_color': badge_color,
                'topic': "General ML",
                'subtopic': template_obj['type'],
                'hint': hint,
                'explanation': explanation
            })
            
        return questions

    def generate_short_questions(self, text, num_questions=3):
        """Generates short answer questions."""
        if not text: return []
        sentences = self.preprocessor.get_sentences(text)
        
        # Simple extraction
        words = re.findall(r'\w+', text.lower())
        c = Counter(words)
        common = [w for w, n in c.most_common(20) if len(w) > 4]
        
        questions = []
        for i in range(num_questions):
            if not common: break
            keyword = random.choice(common)
            common.remove(keyword)
            
            template = random.choice(self.quiz_templates)
            
            if "<OTHER>" in template:
                if common:
                    other = random.choice(common)
                    q_text = template.replace("<KEYWORD>", keyword).replace("<OTHER>", other)
                else:
                    q_text = f"Define {keyword}."
            else:
                q_text = template.replace("<KEYWORD>", keyword)
                
            questions.append({
                'id': i+1,
                'question': q_text,
                'type': 'Short Answer'
            })
        return questions

class CodingRecommender:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        # Helper: Predefined problems
        self.problems = [
             {"id": 1, "title": "Two Sum", "desc": "Find indices of two numbers that add up to target.", "tags": "arrays hash-map loops"},
             {"id": 2, "title": "Reverse String", "desc": "Reverse the input string.", "tags": "string two-pointers"},
             {"id": 3, "title": "Binary Search", "desc": "Search target in sorted array.", "tags": "arrays binary-search logn"},
             {"id": 4, "title": "Fibonacci Number", "desc": "Calculate the Nth fibonacci number.", "tags": "recursion dynamic-programming"},
             {"id": 5, "title": "Bubble Sort", "desc": "Sort an array using bubble sort.", "tags": "sorting arrays loops"}
        ]
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        # Coding Templates
        self.coding_templates = {
            "easy": "Write a simple program using <KEYWORD>.",
            "medium": "Write a program using <KEYWORD> to solve a real-world problem.",
            "hard": "Write an optimized program using <KEYWORD> and analyze its time complexity.",
            "algorithmic": "Write an algorithm and program to implement <KEYWORD>."
        }
        
    def suggest_problems(self, topic):
        """Recommends problems from DB."""
        if not topic: return self.problems[:3]
        
        corpus = [topic] + [f"{p['title']} {p['desc']} {p['tags']}" for p in self.problems]
        try:
            tfidf_matrix = self.tfidf.fit_transform(corpus)
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            related_indices = cosine_sim.argsort()[::-1][:3]
            return [self.problems[i] for i in related_indices]
        except:
            return self.problems[:3]

    def generate_challenge(self, topic, difficulty="medium"):
        """Generates a dynamic coding challenge."""
        template = self.coding_templates.get(difficulty.lower(), self.coding_templates["medium"])
        
        # basic cleanup for keyword injection
        keyword = topic.split()[0].capitalize() if topic else "Python"
        
        question_text = template.replace("<KEYWORD>", keyword)
        return {
            "title": f"{difficulty.capitalize()} Challenge: {keyword}",
            "desc": question_text,
            "tags": f"{difficulty} {keyword.lower()}",
            "generated": True
        }

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

class AITutor:
    """AI Tutor using TF-IDF similarity for RAG context retrieval."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
            
        self.static_kb = [
            {"q": "What is Python?", "a": "**Python** is a high-level, interpreted programming language known for its readability and versatility. It supports multiple paradigms including OOP, functional, and procedural programming. Used in web dev (Django, Flask), data science (pandas, NumPy), ML (TensorFlow, PyTorch), and automation."},
            {"q": "Explain Recursion.", "a": "**Recursion** is when a function calls itself to solve smaller sub-problems. Every recursive function needs a **base case** (stopping condition) and a **recursive case**. Example: `factorial(n) = n * factorial(n-1)` with base case `factorial(0) = 1`. Watch out for stack overflow with deep recursion."},
            {"q": "What is SQL?", "a": "**SQL** (Structured Query Language) manages relational databases. Key commands: `SELECT` (query), `INSERT` (add), `UPDATE` (modify), `DELETE` (remove), `JOIN` (combine tables). Example: `SELECT name, age FROM students WHERE grade > 90 ORDER BY name;`"},
            {"q": "What is a Loop?", "a": "A **loop** repeats code while a condition is true. Types: **for loop** (iterate over a sequence), **while loop** (repeat while condition holds), **do-while** (execute at least once). Example: `for i in range(5): print(i)` prints 0-4."},
            {"q": "Difference between List and Tuple?", "a": "**Lists** are mutable (can be changed after creation) using `[]`. **Tuples** are immutable (cannot be changed) using `()`. Tuples are faster and use less memory. Use tuples for fixed data like coordinates `(x, y)`, lists for collections that change."},
            {"q": "What is Object-Oriented Programming?", "a": "**OOP** organizes code into objects containing data (attributes) and behavior (methods). Four pillars: **Encapsulation** (bundling data+methods), **Inheritance** (child classes extend parent), **Polymorphism** (same interface, different behavior), **Abstraction** (hiding complexity)."},
            {"q": "What is Big O Notation?", "a": "**Big O** describes algorithm efficiency. Common complexities: **O(1)** constant, **O(log n)** logarithmic (binary search), **O(n)** linear, **O(n log n)** (merge sort), **O(n^2)** quadratic (bubble sort), **O(2^n)** exponential. Always aim for lower complexity."},
            {"q": "Explain Stack.", "a": "A **Stack** follows **LIFO** (Last In, First Out). Operations: `push()` (add to top), `pop()` (remove from top), `peek()` (view top). Used in: function call stack, undo/redo, expression evaluation, backtracking algorithms."},
            {"q": "Explain Queue.", "a": "A **Queue** follows **FIFO** (First In, First Out). Operations: `enqueue()` (add to back), `dequeue()` (remove from front). Variants: **Priority Queue**, **Deque** (double-ended). Used in: BFS, task scheduling, print queues."},
            {"q": "What is BFS?", "a": "**BFS** (Breadth-First Search) explores a graph level by level using a **Queue**. It finds the shortest path in unweighted graphs. Time: O(V+E), Space: O(V). Used in: shortest path, social network analysis, web crawling."},
            {"q": "What is DFS?", "a": "**DFS** (Depth-First Search) explores as deep as possible before backtracking, using a **Stack** or recursion. Time: O(V+E), Space: O(V). Used in: topological sorting, cycle detection, maze solving, connected components."},
            {"q": "What is a Binary Tree?", "a": "A **Binary Tree** is a tree where each node has at most 2 children (left and right). A **Binary Search Tree (BST)** maintains order: left < parent < right. Operations (balanced): Search O(log n), Insert O(log n), Delete O(log n)."},
            {"q": "What is a Hash Table?", "a": "A **Hash Table** stores key-value pairs using a hash function to compute indices. Average operations are **O(1)** for insert, search, delete. Handles collisions via chaining or open addressing. Python's `dict` is a hash table."},
            {"q": "What is Dynamic Programming?", "a": "**Dynamic Programming (DP)** solves complex problems by breaking them into overlapping subproblems. Two approaches: **Top-down** (memoization with recursion) and **Bottom-up** (tabulation with iteration). Classic examples: Fibonacci, Knapsack, Longest Common Subsequence."},
            {"q": "What is an API?", "a": "An **API** (Application Programming Interface) lets different software systems communicate. **REST APIs** use HTTP methods: GET (read), POST (create), PUT (update), DELETE (remove). Data is usually exchanged as JSON. Example: fetching weather data from a web service."},
            {"q": "What is Git?", "a": "**Git** is a distributed version control system. Key commands: `git init`, `git add`, `git commit`, `git push`, `git pull`, `git branch`, `git merge`. It tracks changes, enables collaboration, and maintains project history. GitHub/GitLab host remote repos."},
            {"q": "What is a Database?", "a": "A **Database** is an organized collection of data. Types: **Relational** (SQL - MySQL, PostgreSQL) use tables with rows/columns, **NoSQL** (MongoDB, Redis) use documents/key-value pairs. Choose relational for structured data, NoSQL for flexibility and scale."},
            {"q": "What is Machine Learning?", "a": "**Machine Learning** enables computers to learn from data without explicit programming. Types: **Supervised** (labeled data - classification, regression), **Unsupervised** (unlabeled - clustering, dimensionality reduction), **Reinforcement** (reward-based learning)."},
            {"q": "What is HTML CSS JavaScript?", "a": "**HTML** provides structure (headings, paragraphs, forms), **CSS** adds styling (colors, layouts, animations), **JavaScript** adds interactivity (event handling, DOM manipulation, API calls). Together they form the foundation of web development."},
            {"q": "What is a Linked List?", "a": "A **Linked List** stores elements in nodes, each pointing to the next. Types: **Singly** (one direction), **Doubly** (both directions), **Circular** (last points to first). Insert/Delete at head: O(1), Search: O(n). Better than arrays for frequent insertions."},
            {"q": "What are Data Types in Python?", "a": "Python data types: **int** (integers), **float** (decimals), **str** (strings), **bool** (True/False), **list** (mutable sequence), **tuple** (immutable sequence), **dict** (key-value pairs), **set** (unique elements). Python is dynamically typed."},
            {"q": "What is Sorting?", "a": "**Sorting** arranges elements in order. Algorithms: **Bubble Sort** O(n^2), **Selection Sort** O(n^2), **Insertion Sort** O(n^2), **Merge Sort** O(n log n), **Quick Sort** O(n log n avg), **Heap Sort** O(n log n). Merge/Quick sort are preferred for large datasets."},
            {"q": "What is a Graph?", "a": "A **Graph** consists of vertices (nodes) and edges (connections). Types: **Directed/Undirected**, **Weighted/Unweighted**, **Cyclic/Acyclic**. Representations: **Adjacency Matrix** or **Adjacency List**. Used in maps, social networks, web pages."},
            {"q": "What is Inheritance in Python?", "a": "**Inheritance** lets a child class inherit attributes/methods from a parent class. `class Dog(Animal):` means Dog inherits from Animal. Types: **Single**, **Multiple**, **Multilevel**. Use `super()` to call parent methods. Promotes code reuse."},
            {"q": "What is an Array?", "a": "An **Array** stores elements of the same type in contiguous memory. Access by index: O(1). Insert/Delete: O(n) due to shifting. In Python, use `list` (dynamic) or `array` module (typed). NumPy arrays are optimized for numerical computation."},
            {"q": "What is Flask?", "a": "**Flask** is a lightweight Python web framework. Key concepts: **Routes** (`@app.route`), **Templates** (Jinja2), **Request/Response** handling. Extensions: Flask-SQLAlchemy (database), Flask-Login (auth), Flask-Migrate (migrations). Great for APIs and small-medium web apps."},
            {"q": "What is Exception Handling?", "a": "**Exception Handling** manages runtime errors gracefully. Python: `try` (risky code), `except` (handle error), `else` (no error), `finally` (always runs). Example: `try: x = 1/0 except ZeroDivisionError: print('Cannot divide by zero')`. Never use bare `except:`."},
            {"q": "What is Polymorphism?", "a": "**Polymorphism** means 'many forms' - same interface, different behavior. In Python: **Method Overriding** (child redefines parent method), **Duck Typing** (if it walks like a duck...). Example: `len()` works on strings, lists, and dicts differently."},
            {"q": "What is a Class and Object?", "a": "A **Class** is a blueprint defining attributes and methods. An **Object** is an instance of a class. Example: `class Car:` defines the blueprint, `my_car = Car()` creates an object. `__init__` is the constructor that initializes attributes."},
            {"q": "What is Encapsulation?", "a": "**Encapsulation** bundles data and methods together, restricting direct access to internal state. In Python: `_single_underscore` (convention for protected), `__double_underscore` (name mangling for private). Use **getters/setters** or `@property` decorator."},
            {"q": "What is Abstraction?", "a": "**Abstraction** hides complex implementation details and shows only essential features. In Python, use **Abstract Base Classes** (`from abc import ABC, abstractmethod`). Example: A `Shape` class with abstract method `area()` - each shape implements its own calculation."},
        ]
        self.documents = []
        self.answers = []
        
    def build_knowledge_base(self, user_data):
        self.documents = []
        self.answers = []
        
        # 1. Add Static KB
        for item in self.static_kb:
            self.documents.append(item['q'] + " " + item['a']) 
            self.answers.append(item['a'])
            
        # 2. Add User Notes
        if 'notes' in user_data:
            for note in user_data['notes']:
                self.documents.append(note)
                self.answers.append(f"From your notes: {note[:200]}...") 
        
        # 3. Add User Code
        if 'code' in user_data:
            for code in user_data['code']:
                self.documents.append(code)
                self.answers.append(f"From your code submission: \n{code[:100]}...")

        # 4. Add Syllabus
        if 'syllabus' in user_data:
            for syl in user_data['syllabus']:
                self.documents.append(syl)
                self.answers.append(f"From your study plan: {syl}")
                
        if self.documents:
            try:
                self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            except ValueError:
                self.tfidf_matrix = None

    def get_response(self, query, history=[], api_key=None, model_type="fast"):
        """Get AI tutor response — Gemini API first, local RAG as fallback."""

        # --- 1. Build local RAG context (used to enrich the Gemini prompt) ---
        context = ""
        context_query = query
        if history:
            last_msg = history[-1] if history else ""
            context_query = f"{last_msg} {query}"

        if getattr(self, 'tfidf_matrix', None) is not None:
            try:
                query_vec = self.vectorizer.transform([context_query])
                similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
                best_idx = np.argmax(similarities)
                if similarities[best_idx] > 0.15:
                    context = self.answers[best_idx]
            except Exception:
                pass

        # --- 2. Build conversation history string ---
        history_str = ""
        if history:
            turns = []
            for i in range(0, len(history) - 1, 2):
                if i + 1 < len(history):
                    turns.append(f"Student: {history[i]}\nTutor: {history[i+1]}")
            if turns:
                history_str = "\n".join(turns[-3:])  # Last 3 turns

        # --- 3. External API Call (Groq or Gemini) ---
        if not getattr(AITutor, '_api_dead', False) and api_key and len(api_key) > 10:
            try:
                # Build prompt based on mode
                prompt_parts = [
                    "You are SmartRevise AI Tutor - an expert academic tutor.",
                    "Use markdown formatting: **bold** for key terms, `code` for inline code, ```lang for code blocks.",
                ]
                if model_type == "think":
                    prompt_parts.append("Think deeply and step-by-step. Provide a comprehensive, highly detailed response with thorough explanations, edge cases, and complete code examples with comments.")
                else:
                    prompt_parts.append("Keep answers concise and focused (2-4 paragraphs max). Be direct and efficient.")

                if context:
                    prompt_parts.append(f"\nContext from student's notes:\n{context}")
                if history_str:
                    prompt_parts.append(f"\nRecent conversation:\n{history_str}")
                prompt_parts.append(f"\nStudent's question: {query}")
                full_prompt = "\n".join(prompt_parts)

                # Route based on API key prefix
                if api_key.startswith("gsk_"):
                    from groq import Groq
                    client = Groq(api_key=api_key)
                    groq_model = "llama3-70b-8192" if model_type == "think" else "llama3-8b-8192"
                    completion = client.chat.completions.create(
                        model=groq_model,
                        messages=[{"role": "user", "content": full_prompt}]
                    )
                    return completion.choices[0].message.content

                else:
                    # Use Gemini API — Fast: flash-lite, Think: flash (deeper reasoning)
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    gemini_model = 'gemini-3-flash-preview' if model_type == "think" else 'gemini-3.1-flash-lite-preview'
                    model = genai.GenerativeModel(gemini_model)
                    response = model.generate_content(full_prompt)
                    if response.text:
                        return response.text

            except Exception as e:
                print("API Error:", e)
                if "429" in str(e) or "quota" in str(e).lower():
                    AITutor._api_dead = True
                    print("  [!] API quota exhausted -- using local KB")

        # --- 4. Local Fallback (if no API key or Gemini unavailable) ---
        if context:
            return context

        # Static KB direct match
        query_lower = query.lower()
        for item in self.static_kb:
            if any(word in query_lower for word in item['q'].lower().split() if len(word) > 3):
                return item['a']

        if getattr(AITutor, '_api_dead', False):
            return "My Gemini API quota has been exhausted. I am currently in local fallback mode and can only answer questions directly related to your notes or basic pre-programmed concepts (like Stack, Queue, API, Machine Learning)."
            
        return "I need a valid Gemini API key to answer this question, or my current key has exceeded its rate limits. Please configure a new Gemini API key in the app settings."

class AnalyticsEngine:
    """
    Advanced Analytics for Student Performance:
    1. Knowledge Gap Detector
    2. Mistake Pattern Analyzer
    3. Study Path Generator (Personalized)
    """
    def compute_analytics(self, questions, user_answers, time_taken_map=None):
        analysis = {
            "weak_concepts": [],
            "mistake_patterns": [],
            "study_path": [],
            "summary": ""
        }
        
        incorrect_topics = []
        topic_performance = {} # {topic: {correct: 0, total: 0}}
        mistake_counts = {"Conceptual": 0, "Syntax": 0, "Time": 0, "Guessing": 0}
        
        for q in questions:
            qid = q['id']
            selected = user_answers.get(f'q{qid}')
            correct = q['answer']
            topic = q.get('topic', 'General')
            
            if topic not in topic_performance: topic_performance[topic] = {'correct': 0, 'total': 0}
            topic_performance[topic]['total'] += 1
            
            if selected == correct:
                topic_performance[topic]['correct'] += 1
            else:
                # 1. Knowledge Gap Detection
                subtopic = q.get('subtopic', 'Concept')
                incorrect_topics.append(f"{topic}: {subtopic}")
                
                # 2. Mistake Pattern Analysis
                time = 0
                if time_taken_map:
                    time = float(time_taken_map.get(str(qid), 15))
                
                q_type = q.get('subtopic', '')
                
                if time < 5:
                    mistake_counts["Guessing"] += 1
                elif "Output" in q_type or "Syntax" in q_type:
                    mistake_counts["Syntax"] += 1
                elif time > 45:
                    mistake_counts["Conceptual"] += 1
                else: 
                     mistake_counts["Conceptual"] += 1

        # Aggregate Gaps
        if incorrect_topics:
            c = Counter(incorrect_topics)
            analysis["weak_concepts"] = [topic for topic, count in c.most_common(3)]
        
        # Aggregate Patterns
        dominant_mistake = max(mistake_counts, key=mistake_counts.get)
        if mistake_counts[dominant_mistake] > 0:
            analysis["mistake_patterns"].append({
                "type": f"{dominant_mistake} Error",
                "count": mistake_counts[dominant_mistake],
                "suggestion": self._get_suggestion(dominant_mistake)
            })

        # 3. Personalized Study Path Generator
        # Logic based on Topic Mastery
        path = []
        
        # Identify weakest topic
        weakest_topic = None
        min_acc = 1.0
        for t, data in topic_performance.items():
            acc = data['correct'] / data['total']
            if acc < min_acc:
                min_acc = acc
                weakest_topic = t
        
        if weakest_topic and min_acc < 0.7:
            # Plan for weak topic
            path.append({"label": "Today", "topic": f"{weakest_topic} Basics", "action": "Review core definitions and concepts."})
            path.append({"label": "Next", "topic": f"{weakest_topic} Applied", "action": "Practice syntax and simple problems."})
            path.append({"label": "Later", "topic": "Related Topics", "action": "Move to advanced subtopics."})
        elif analysis["weak_concepts"]:
             # Plan based on specific gaps
            focus = analysis["weak_concepts"][0].split(':')[0]
            path.append({"label": "Today", "topic": focus, "action": "Deep dive into this weak area."})
            path.append({"label": "Next", "topic": "Mixed Practice", "action": "Take a mixed bag quiz."})
            path.append({"label": "Later", "topic": "Advanced Concepts", "action": "Challenge yourself with Hard questions."})
        else:
            # Mastery Plan
            path.append({"label": "Today", "topic": "Advanced Application", "action": "You mastered the basics! Try coding challenges."})
            path.append({"label": "Next", "topic": "New Subject", "action": "Start learning a new module."})
            path.append({"label": "Later", "topic": "Project Building", "action": "Apply concepts in a real project."})
            
        analysis["study_path"] = path
        return analysis

    def _get_suggestion(self, mistake_type):
        suggestions = {
            "Conceptual": "Review the core concepts and definitions.",
            "Syntax": "Practice writing code snippets without an IDE.",
            "Time": "Try timed quizzes to improve your speed.",
            "Guessing": "Don't guess! Review the topic if unsure."
        }
        return suggestions.get(mistake_type, "Review your wrong answers.")

    def generate_study_plan_from_text(self, text):
        """Generates a study plan from syllabus text."""
        if not text: return []
        
        # Simple extraction of potential topics (capitalized words or lines)
        lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 5]
        
        # If text is paragraph, extract key phrases
        if len(lines) < 3:
            # Fallback to keyphrase extraction
            import re
            words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
            c = Counter(words)
            topics = [w for w, n in c.most_common(10) if len(w) > 3]
        else:
            topics = lines[:10]
            
        if not topics:
            topics = ["General Concepts", "Key Definitions", "Advanced Applications"]
            
        plan = []
        
        # Distribute topics
        # Today: First 30%
        # Next: Next 40%
        # Later: Last 30%
        
        n = len(topics)
        t1 = max(1, int(n * 0.3))
        t2 = max(1, int(n * 0.7))
        
        for i, topic in enumerate(topics):
            if i < t1:
                label = "Today"
                action = "Read definitions and understand the scope."
            elif i < t2:
                label = "Next"
                action = "Practice questions and application."
            else:
                label = "Later"
                action = "Review advanced concepts and edge cases."
                
            plan.append({"label": label, "topic": topic, "action": action})
            
        return plan
