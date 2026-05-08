"""
PDF Intelligence Engine — Advanced Traditional ML
Algorithms: TextRank (PageRank), TF-IDF, SpaCy NER, KMeans, WordNet
No deep learning — fast, accurate, offline-capable.
"""

import re
import random
import math
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from model_cache import cache as _cache


# ── helpers ──────────────────────────────────────────────
def _nlp():
    return _cache.get_spacy()

def _nltk_ok():
    return _cache.is_nltk_ready()


# ════════════════════════════════════════════════════════
# 1. PDF TEXT EXTRACTION
# ════════════════════════════════════════════════════════
class PDFExtractor:
    def extract(self, file_stream):
        try:
            import PyPDF2, io
            if hasattr(file_stream, 'seek'):
                file_stream.seek(0)
            raw = file_stream.read() if hasattr(file_stream, 'read') else file_stream
            if not raw or len(raw) < 10:
                return "Error: Empty or corrupt PDF.", 0
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            pages = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    pages.append(t.strip())
            if not pages:
                return "Error: No text extracted. May be a scanned/image PDF.", 0
            return self._clean("\n\n".join(pages)), len(reader.pages)
        except Exception as e:
            return f"Error: {e}", 0

    def _clean(self, text):
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)   # fix hyphenation
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\s*•\s*', '\n• ', text)
        return text.strip()


# ════════════════════════════════════════════════════════
# 2. TEXT PREPROCESSOR
# ════════════════════════════════════════════════════════
STOP = {
    "the","is","at","which","on","and","a","an","in","to","of","for","with",
    "as","by","that","this","it","or","are","be","from","was","were","but",
    "not","have","has","can","will","if","than","then","we","you","they","he",
    "she","do","does","did","would","could","should","its","also","been",
    "being","each","more","most","other","some","such","only","own","so",
    "about","into","over","after","before","between","under","again","there",
    "here","when","where","how","all","both","through","very","just","may",
}

class TextProcessor:
    def sentences(self, text):
        """SpaCy sentence splitting with regex fallback."""
        nlp = _nlp()
        if nlp and len(text) < 500_000:
            try:
                doc = nlp(text[:500_000])
                sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]
                if sents:
                    return sents
            except Exception:
                pass
        # regex fallback
        raw = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in raw if len(s.strip()) > 20]

    def tokens(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).split()

    def keywords(self, text, top_n=20):
        sents = self.sentences(text)
        if len(sents) < 2:
            words = [w for w in self.tokens(text) if w not in STOP and len(w) > 2]
            return [w for w, _ in Counter(words).most_common(top_n)]
        try:
            tf = TfidfVectorizer(stop_words='english', max_features=200, ngram_range=(1,2))
            mat = tf.fit_transform(sents)
            scores = np.asarray(mat.sum(axis=0)).flatten()
            names = tf.get_feature_names_out()
            ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
            return [w for w, _ in ranked[:top_n]]
        except Exception:
            words = [w for w in self.tokens(text) if w not in STOP and len(w) > 2]
            return [w for w, _ in Counter(words).most_common(top_n)]


# ════════════════════════════════════════════════════════
# 3. TEXTRANK SUMMARIZER (PageRank on sentence graph)
# ════════════════════════════════════════════════════════
class TextRankSummarizer:
    """
    True TextRank: build TF-IDF sentence similarity graph,
    run PageRank, return top-ranked sentences in document order.
    """

    def __init__(self):
        self.proc = TextProcessor()

    def summarize(self, text, max_points=6, api_key=None):
        if not text or len(text.strip()) < 100:
            return ["Text too short to summarize."]

        # --- ADVANCED: Gemini abstractive summary ---
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = (
                    "Summarize the following text into clear, high-impact bullet points for a student.\n"
                    f"Text:\n{text[:30000]}\n\n"
                    f"Provide exactly {max_points} bullet points. Focus on key takeaways."
                )
                response = model.generate_content(prompt)
                points = [p.strip().lstrip('*-• ') for p in response.text.split('\n') if p.strip()]
                if points:
                    return points[:max_points]
            except Exception as e:
                print(f"[Intelligence] Gemini Summarization Error: {e}")

        # --- TRADITIONAL: Fallback to TextRank ---
        sents = self.proc.sentences(text)
        if len(sents) < 3:
            return sents[:max_points]

        # --- TF-IDF similarity matrix ---
        try:
            tf = TfidfVectorizer(stop_words='english')
            mat = tf.fit_transform(sents)
            sim = cosine_similarity(mat)
        except Exception:
            return self._fallback(sents, text, max_points)

        # --- PageRank on similarity graph ---
        scores = self._pagerank(sim)

        # --- Positional bias (first 20% of doc is important) ---
        n = len(sents)
        for i in range(n):
            if i < max(3, n // 5):
                scores[i] *= 1.25
            # boost definition sentences
            lower = sents[i].lower()
            if any(p in lower for p in [" is a ", " is an ", " refers to ", " defined as ", " means "]):
                scores[i] *= 1.3

        # --- Select top sentences, keep original order ---
        ranked_idx = sorted(range(n), key=lambda i: scores[i], reverse=True)
        top_idx = sorted(ranked_idx[:max_points * 2])  # keep doc order

        # --- Deduplicate similar sentences ---
        selected = []
        for i in top_idx:
            if len(selected) >= max_points:
                break
            sent = sents[i]
            # skip if too similar to already-selected
            if selected:
                try:
                    tf2 = TfidfVectorizer(stop_words='english')
                    vecs = tf2.fit_transform(selected + [sent])
                    sims = cosine_similarity(vecs[-1], vecs[:-1]).flatten()
                    if sims.max() > 0.65:
                        continue
                except Exception:
                    pass
            selected.append(sent)

        return selected if selected else sents[:max_points]

    def _pagerank(self, sim_matrix, damping=0.85, iterations=50):
        n = len(sim_matrix)
        # normalise rows
        row_sums = sim_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        norm = sim_matrix / row_sums
        scores = np.ones(n) / n
        for _ in range(iterations):
            scores = (1 - damping) / n + damping * norm.T.dot(scores)
        return scores

    def _fallback(self, sents, text, max_points):
        proc = self.proc
        kws = set(proc.keywords(text, top_n=20))
        scored = []
        for i, s in enumerate(sents):
            score = sum(1 for w in proc.tokens(s) if w in kws)
            if i < 3:
                score += 2
            scored.append((s, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:max_points]]


# ════════════════════════════════════════════════════════
# 4. KEY POINTS EXTRACTOR (NER + TF-IDF hybrid)
# ════════════════════════════════════════════════════════
class KeyPointExtractor:
    NER_LABELS = {"ORG","PRODUCT","EVENT","PERSON","GPE","LAW","NORP","WORK_OF_ART","FAC"}

    def __init__(self):
        self.proc = TextProcessor()
        self.summarizer = TextRankSummarizer()

    def extract(self, text, max_points=8):
        if not text:
            return []
        sents = self.proc.sentences(text)
        nlp = _nlp()
        scores = defaultdict(float)

        # NER boost — sentences containing named entities
        if nlp:
            try:
                doc = nlp(text[:200_000])
                entity_sents = set()
                for ent in doc.ents:
                    if ent.label_ in self.NER_LABELS:
                        entity_sents.add(ent.sent.text.strip())
                for s in sents:
                    if s in entity_sents:
                        scores[s] += 4.0
            except Exception:
                pass

        # TF-IDF keyword score
        kws = set(self.proc.keywords(text, top_n=25))
        for s in sents:
            for w in self.proc.tokens(s):
                if w in kws:
                    scores[s] += 1.0
            lower = s.lower()
            if any(p in lower for p in [" is ", " defined ", " means ", " refers "]):
                scores[s] += 2.0
            if any(p in lower for p in ["important","key","significant","essential","critical","main"]):
                scores[s] += 1.5

        ranked = sorted(sents, key=lambda s: scores[s], reverse=True)

        # Deduplicate
        selected = []
        for s in ranked:
            if len(selected) >= max_points:
                break
            if not selected:
                selected.append(s)
                continue
            try:
                tf = TfidfVectorizer(stop_words='english')
                vecs = tf.fit_transform(selected + [s])
                sims = cosine_similarity(vecs[-1], vecs[:-1]).flatten()
                if sims.max() < 0.60:
                    selected.append(s)
            except Exception:
                selected.append(s)

        return selected[:max_points]


# ════════════════════════════════════════════════════════
# 5. SHORT NOTES GENERATOR (Topic Clustering + NER)
# ════════════════════════════════════════════════════════
class ShortNotesGenerator:
    """
    Generates structured notes by:
    1. Extracting definitions, facts, concepts by pattern
    2. Clustering sentences by topic with KMeans + TF-IDF
    3. Using SpaCy NER for entity-based facts
    """

    def __init__(self):
        self.proc = TextProcessor()

    def generate(self, text, api_key=None):
        if not text:
            return {}
        
        # --- ADVANCED: Gemini structured notes ---
        if api_key:
            try:
                import google.generativeai as genai
                import json
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = (
                    "Analyze the following text and generate structured study notes in JSON format.\n"
                    "JSON Schema: {\"definitions\": [], \"key_concepts\": [], \"important_facts\": [], \"important_terms\": [], \"summary\": \"\"}\n\n"
                    f"Text:\n{text[:20000]}"
                )
                response = model.generate_content(prompt)
                # Extract JSON from response (handling potential markdown formatting)
                raw = response.text
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0].strip()
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0].strip()
                
                data = json.loads(raw)
                # Ensure all keys exist
                for k in ["definitions", "key_concepts", "important_facts", "important_terms"]:
                    if k not in data: data[k] = []
                return data
            except Exception as e:
                print(f"[Intelligence] Gemini Notes Error: {e}")

        # --- TRADITIONAL: Fallback ---
        sents = self.proc.sentences(text)
        kws = self.proc.keywords(text, top_n=12)
        notes = {
            "definitions": [],
            "key_concepts": [],
            "important_facts": [],
            "important_terms": kws[:10],
            "topic_clusters": []
        }

        DEF_PATTERNS = [" is a ", " is an ", " refers to ", " defined as ", " means ", " is defined ", " can be defined "]
        CONCEPT_PATTERNS = ["important", "key", "significant", "essential", "critical", "main", "primary", "fundamental"]
        FACT_PATTERNS = [" was ", " were ", " discovered ", " invented ", " founded ", " first ", " developed by ", " proposed "]

        for s in sents:
            lower = s.lower()
            if any(p in lower for p in DEF_PATTERNS) and 20 < len(s) < 300:
                notes["definitions"].append(s)
            elif any(p in lower for p in CONCEPT_PATTERNS) and 20 < len(s) < 300:
                notes["key_concepts"].append(s)
            elif any(p in lower for p in FACT_PATTERNS) and 20 < len(s) < 300:
                notes["important_facts"].append(s)

        # NER-based facts
        nlp = _nlp()
        if nlp:
            try:
                doc = nlp(text[:200_000])
                seen = set()
                for ent in doc.ents:
                    if ent.label_ in {"PERSON","ORG","EVENT","LAW"} and len(ent.text) > 2:
                        fact = f"{ent.text} ({ent.label_}): {ent.sent.text.strip()}"
                        if fact not in seen and len(fact) < 350:
                            seen.add(fact)
                            notes["important_facts"].append(fact)
            except Exception:
                pass

        # Topic clustering via KMeans + TF-IDF
        notes["topic_clusters"] = self._cluster_topics(sents, text)

        # Trim
        notes["definitions"]    = notes["definitions"][:6]
        notes["key_concepts"]   = notes["key_concepts"][:6]
        notes["important_facts"] = notes["important_facts"][:6]

        return notes

    def _cluster_topics(self, sents, text, n_clusters=4):
        """Group sentences by topic and name each cluster."""
        if len(sents) < n_clusters * 2:
            return []
        try:
            tf = TfidfVectorizer(stop_words='english', max_features=300)
            mat = tf.fit_transform(sents)
            n = min(n_clusters, len(sents) // 2)
            km = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = km.fit_predict(mat)
            feature_names = tf.get_feature_names_out()

            clusters = []
            for c in range(n):
                idx = [i for i, l in enumerate(labels) if l == c]
                if not idx:
                    continue
                # Top terms = cluster heading
                centroid = km.cluster_centers_[c]
                top_term_idx = centroid.argsort()[-3:][::-1]
                heading = ", ".join(feature_names[i].title() for i in top_term_idx)
                # Best sentence per cluster (highest TF-IDF sum)
                cluster_mat = mat[idx]
                row_scores = np.asarray(cluster_mat.sum(axis=1)).flatten()
                best = sents[idx[row_scores.argmax()]]
                clusters.append({"topic": heading, "summary": best})
            return clusters
        except Exception:
            return []


# ════════════════════════════════════════════════════════
# 6. ADVANCED MCQ GENERATOR
# ════════════════════════════════════════════════════════
class PDFMCQGenerator:
    """
    Generates high-quality MCQs using:
    - SpaCy NER for entity extraction
    - Noun chunk extraction
    - Context sentence as answer explanation
    - Multiple question templates per type
    - Semantic distractor generation (same NER type + TF-IDF similar)
    """

    Q_TEMPLATES = {
        "definition": [
            "What is '{keyword}'?",
            "Which of the following best describes '{keyword}'?",
            "How is '{keyword}' defined in the given content?",
        ],
        "entity": [
            "What role does '{keyword}' play according to the content?",
            "Which statement about '{keyword}' is most accurate?",
            "What is '{keyword}' primarily associated with?",
        ],
        "concept": [
            "Which of the following correctly explains '{keyword}'?",
            "What does '{keyword}' refer to in this context?",
            "'{keyword}' is best described as which of the following?",
        ],
        "fill": [
            "Complete: '... {blank} ...' — which term fits the blank?",
        ]
    }

    def __init__(self):
        self.proc = TextProcessor()

    def generate(self, text, num_questions=10, difficulty="medium", api_key=None):
        if not text:
            return []

        # --- ADVANCED: Gemini high-accuracy MCQs ---
        if api_key:
            try:
                import google.generativeai as genai
                import json
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = (
                    f"Generate {num_questions} high-quality Multiple Choice Questions at '{difficulty}' difficulty level based on the text below.\n"
                    "For each question, provide 1 correct answer and 3 highly plausible but incorrect distractors.\n"
                    "JSON Format: [{\"question\": \"\", \"options\": [\"\", \"\", \"\", \"\"], \"answer\": \"\", \"explanation\": \"\", \"topic\": \"\", \"subtopic\": \"\", \"hint\": \"\"}]\n\n"
                    f"Text:\n{text[:20000]}"
                )
                response = model.generate_content(prompt)
                raw = response.text
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0].strip()
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0].strip()
                
                questions = json.loads(raw)
                # Post-process for app format
                for i, q in enumerate(questions):
                    q['id'] = i + 1
                    q['badge_color'] = {"easy": "success", "hard": "danger"}.get(difficulty, "warning")
                    q['difficulty'] = difficulty.capitalize()
                    random.shuffle(q['options'])
                return questions
            except Exception as e:
                print(f"[Intelligence] Gemini MCQ Error: {e}")

        # --- TRADITIONAL: Fallback ---
        _nltk_ok()
        sents = self.proc.sentences(text)
        candidates = self._extract_candidates(text, sents)

        if len(candidates) < 4:
            return []

        random.shuffle(candidates)
        questions = []
        used_kws = set()

        for kw, sent, qtype in candidates:
            if len(questions) >= num_questions:
                break
            norm = kw.lower().strip()
            if norm in used_kws or len(norm) < 2:
                continue

            distractors = self._distractors(kw, qtype, candidates, text)
            if len(distractors) < 3:
                continue

            q_text = self._make_question(kw, sent, qtype)
            options = [kw] + distractors[:3]
            random.shuffle(options)
            explanation = self._explanation(kw, sent, sents)
            diff_label, badge = self._diff(difficulty)

            questions.append({
                "id": len(questions) + 1,
                "question": q_text,
                "options": options,
                "answer": kw,
                "difficulty": diff_label,
                "badge_color": badge,
                "topic": "PDF Analysis",
                "subtopic": qtype.replace("_", " ").title(),
                "hint": f"Hint: Look in context — '{sent[:70].strip()}...'",
                "explanation": explanation
            })
            used_kws.add(norm)

        return questions

    # ── private helpers ──────────────────────────────────

    def _extract_candidates(self, text, sents):
        candidates = []
        nlp = _nlp()
        GOOD_ENT = {"ORG","PRODUCT","EVENT","PERSON","GPE","LAW","NORP","WORK_OF_ART"}

        if nlp:
            try:
                doc = nlp(text[:200_000])
                for ent in doc.ents:
                    if ent.label_ in GOOD_ENT and len(ent.text.strip()) > 2:
                        candidates.append((ent.text.strip(), ent.sent.text.strip(), "entity"))
                for chunk in doc.noun_chunks:
                    if 1 <= len(chunk.text.split()) <= 4 and len(chunk.text) > 3:
                        sent_text = chunk.sent.text.strip()
                        candidates.append((chunk.text.strip(), sent_text, "concept"))
            except Exception:
                pass

        # Pattern-based: definition sentences → pick defined term
        for s in sents:
            lower = s.lower()
            for pat in [" is a ", " is an ", " refers to ", " defined as "]:
                if pat in lower:
                    # term is what precedes the pattern
                    idx = lower.index(pat)
                    term = s[:idx].split()[-1].strip(",:;")
                    if len(term) > 2:
                        candidates.append((term, s, "definition"))
                    break

        # TF-IDF keyword fallback
        kws = self.proc.keywords(text, top_n=25)
        for kw in kws:
            for s in sents:
                if kw.lower() in s.lower():
                    candidates.append((kw, s, "concept"))
                    break

        # Deduplicate by keyword
        seen = set()
        unique = []
        for kw, s, qt in candidates:
            k = kw.lower().strip()
            if k not in seen:
                seen.add(k)
                unique.append((kw, s, qt))
        return unique

    def _distractors(self, word, qtype, all_candidates, text):
        distractors = []

        # 1. Same-type candidates from PDF
        same_type = [c[0] for c in all_candidates
                     if c[2] == qtype and c[0].lower() != word.lower()]
        distractors.extend(same_type[:4])

        # 2. WordNet hyponyms/synonyms
        if _nltk_ok():
            try:
                from nltk.corpus import wordnet
                syns = wordnet.synsets(word.split()[0])
                if syns:
                    for lemma in syns[0].lemmas():
                        n = lemma.name().replace('_', ' ')
                        if n.lower() != word.lower():
                            distractors.append(n)
                    for hyp in syns[0].hypernyms():
                        for h in hyp.hyponyms():
                            n = h.lemmas()[0].name().replace('_', ' ')
                            if n.lower() != word.lower():
                                distractors.append(n)
            except Exception:
                pass

        # 3. TF-IDF similar terms from all candidates
        all_kws = [c[0] for c in all_candidates if c[0].lower() != word.lower()]
        if len(all_kws) >= 4:
            try:
                pool = [word] + all_kws[:30]
                tf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
                vecs = tf.fit_transform(pool)
                sims = cosine_similarity(vecs[0:1], vecs[1:])[0]
                sorted_idx = sims.argsort()[::-1]
                for i in sorted_idx[:6]:
                    d = all_kws[i]
                    if d.lower() != word.lower():
                        distractors.append(d)
            except Exception:
                pass

        # Clean and deduplicate
        seen = {word.lower()}
        clean = []
        for d in distractors:
            dl = d.lower().strip()
            if dl not in seen and len(dl) > 1:
                seen.add(dl)
                clean.append(d)
            if len(clean) >= 3:
                break
        return clean

    def _make_question(self, kw, sent, qtype):
        templates = self.Q_TEMPLATES.get(qtype, self.Q_TEMPLATES["concept"])
        t = random.choice(templates)
        if '{keyword}' in t:
            return t.format(keyword=kw)
        if '{blank}' in t:
            masked = sent.replace(kw, "_____", 1)
            return t.format(blank=masked[:80])
        return t

    def _explanation(self, kw, sent, all_sents):
        """Return the context sentence + the sentence after it if available."""
        try:
            idx = all_sents.index(sent)
            ctx = sent
            if idx + 1 < len(all_sents):
                ctx += " " + all_sents[idx + 1]
        except ValueError:
            ctx = sent
        return f"'{kw}' — {ctx.strip()}"

    def _diff(self, d):
        return {"easy": ("Easy", "success"),
                "hard": ("Hard", "danger")}.get(d, ("Medium", "warning"))


# ════════════════════════════════════════════════════════
# 7. PDF Q&A (TF-IDF RAG — Gemini-powered)
# ════════════════════════════════════════════════════════
class PDFQuestionAnswerer:
    def __init__(self):
        self.proc = TextProcessor()
        self.chunks = []
        self._tfidf = None
        self._tfidf_mat = None

    def build_index(self, text):
        sents = self.proc.sentences(text)
        self.chunks = []
        for i in range(0, len(sents), 2):
            chunk = " ".join(sents[i:i+3])
            if len(chunk) > 30:
                self.chunks.append(chunk)
        if self.chunks:
            self._tfidf = TfidfVectorizer(stop_words='english')
            self._tfidf_mat = self._tfidf.fit_transform(self.chunks)

    def answer(self, question, api_key=None):
        if not self.chunks:
            return "No PDF indexed. Upload first."
        context = self._retrieve(question)
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = (
                    "You are an expert document analyst. Answer ONLY from the context below.\n\n"
                    f"Context:\n{context}\n\nQuestion: {question}\n\n"
                    "Be concise, accurate, and use bullet points when helpful."
                )
                return model.generate_content(prompt).text
            except Exception as e:
                print(f"[PDF-QA] Gemini error: {e}")
        return f"Based on the PDF:\n\n{context}"

    def _retrieve(self, question, top_k=4):
        if not self._tfidf or self._tfidf_mat is None:
            return "\n\n".join(self.chunks[:top_k])
        try:
            q_vec = self._tfidf.transform([question])
            sims = cosine_similarity(q_vec, self._tfidf_mat).flatten()
            idx = sims.argsort()[-top_k:][::-1]
            relevant = [self.chunks[i] for i in idx if sims[i] > 0.05]
            return "\n\n".join(relevant) if relevant else "\n\n".join(self.chunks[:top_k])
        except Exception:
            return "\n\n".join(self.chunks[:top_k])


# ════════════════════════════════════════════════════════
# 8. MAIN ORCHESTRATOR
# ════════════════════════════════════════════════════════
class PDFIntelligence:
    def __init__(self):
        self.extractor   = PDFExtractor()
        self.summarizer  = TextRankSummarizer()
        self.key_points  = KeyPointExtractor()
        self.notes_gen   = ShortNotesGenerator()
        self.mcq_gen     = PDFMCQGenerator()
        self.qa_engine   = PDFQuestionAnswerer()
        self._text       = ""

    def process_pdf(self, file_stream):
        text, pages = self.extractor.extract(file_stream)
        if text.startswith("Error"):
            return {"error": text, "pages": 0}
        self._text = text
        return {"success": True, "text": text, "pages": pages, "word_count": len(text.split())}

    def get_summary(self, text=None, max_points=6, api_key=None):
        return self.summarizer.summarize(text or self._text, max_points, api_key)

    def get_key_points(self, text=None, max_points=8):
        return self.key_points.extract(text or self._text, max_points)

    def get_short_notes(self, text=None, api_key=None):
        return self.notes_gen.generate(text or self._text, api_key)

    def get_mcqs(self, text=None, num_questions=10, difficulty="medium", api_key=None):
        return self.mcq_gen.generate(text or self._text, num_questions, difficulty, api_key)

    def ask_question(self, question, text=None, api_key=None):
        t = text or self._text
        if t:
            self.qa_engine.build_index(t)
        return self.qa_engine.answer(question, api_key)
