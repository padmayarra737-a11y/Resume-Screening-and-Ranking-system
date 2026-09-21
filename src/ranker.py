"""
Step 4: Score and rank resumes against a job description.

final score = w_sem * semantic similarity
            + w_skill * skill match
            + w_exp * experience match
(default weights 60% / 30% / 10%)

Semantic similarity can use:
  - "tfidf"      : always available (scikit-learn)
  - "embeddings" : sentence-transformers, optional (pip install sentence-transformers)
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.extractor import extract_all, extract_required_years, extract_skills
from src.load_data import clean_text

# TF-IDF cosine values for a good match are usually 0.15-0.3, so we
# rescale to a 0-1 range to make the percentage easier to read.
TFIDF_FULL_MARK = 0.3


def embeddings_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class ResumeRanker:
    def __init__(self, method="tfidf", weights=(0.6, 0.3, 0.1), background_texts=None):
        """
        method           : "tfidf" or "embeddings"
        weights          : (semantic, skills, experience) -- should add up to 1
        background_texts : optional list of extra texts (e.g. the 2,481 dataset
                           resumes) used only to learn word importance (IDF).
                           Helps a lot when ranking just a few uploaded resumes.
        """
        if method not in ("tfidf", "embeddings"):
            raise ValueError("method must be 'tfidf' or 'embeddings'")
        self.method = method
        self.weights = weights
        self.background_texts = background_texts or []
        self._model = None
        self._vec = None
        self._cache = {}   # text -> (clean text, extracted info)

    # ---------- semantic similarity ----------
    def _get_vectorizer(self, extra_texts):
        """With background texts: fit once and reuse (fast, stable word weights).
        Without: fit on the resumes + JD being ranked."""
        make = lambda: TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                       sublinear_tf=True, max_features=60000)
        if self.background_texts:
            if self._vec is None:
                self._vec = make().fit(self.background_texts)
            return self._vec
        return make().fit(extra_texts)

    def _tfidf_similarity(self, jd_clean, resume_cleans):
        vec = self._get_vectorizer(resume_cleans + [jd_clean])
        jd_v = vec.transform([jd_clean])
        res_v = vec.transform(resume_cleans)
        return cosine_similarity(jd_v, res_v).ravel()

    def _embedding_similarity(self, jd_clean, resume_cleans):
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        jd_e = self._model.encode([jd_clean], normalize_embeddings=True)
        res_e = self._model.encode(resume_cleans, normalize_embeddings=True,
                                   batch_size=32, show_progress_bar=False)
        return (res_e @ jd_e.T).ravel()

    def semantic_scores(self, jd_clean, resume_cleans):
        """Raw similarity (0-1) between the JD and each resume."""
        if self.method == "embeddings":
            return self._embedding_similarity(jd_clean, resume_cleans)
        return self._tfidf_similarity(jd_clean, resume_cleans)

    def _process(self, text):
        """Clean + extract once per resume text and remember the result."""
        key = hash(text)
        if key not in self._cache:
            self._cache[key] = (clean_text(text), extract_all(text))
        return self._cache[key]

    # ---------- full ranking ----------
    def rank(self, jd_text, resumes):
        """
        jd_text : job description (raw text)
        resumes : list of dicts  {"name": str, "text": str, ...extra fields kept}
        Returns a DataFrame sorted best -> worst.
        """
        if not resumes:
            return pd.DataFrame()

        jd_clean = clean_text(jd_text)
        processed = [self._process(r["text"]) for r in resumes]
        resume_cleans = [p[0] for p in processed]

        sem_raw = self.semantic_scores(jd_clean, resume_cleans)
        if self.method == "tfidf":
            sem = np.clip(sem_raw / TFIDF_FULL_MARK, 0, 1)
        else:
            sem = np.clip(sem_raw, 0, 1)

        jd_skills = set(extract_skills(jd_text))
        required_years = extract_required_years(jd_text)

        w_sem, w_skill, w_exp = self.weights
        # If the JD has no recognisable skills / years, drop that part
        # and share its weight so scores are not unfairly reduced.
        if not jd_skills:
            w_sem, w_skill = w_sem + w_skill, 0.0
        if required_years == 0:
            w_sem, w_exp = w_sem + w_exp, 0.0
        total = w_sem + w_skill + w_exp
        w_sem, w_skill, w_exp = w_sem / total, w_skill / total, w_exp / total

        rows = []
        for i, r in enumerate(resumes):
            info = processed[i][1]
            res_skills = set(info["skills"])
            matched = sorted(jd_skills & res_skills)
            missing = sorted(jd_skills - res_skills)
            skill_score = len(matched) / len(jd_skills) if jd_skills else 0.0
            exp_score = (min(info["years_experience"] / required_years, 1.0)
                         if required_years else 0.0)

            final = w_sem * sem[i] + w_skill * skill_score + w_exp * exp_score
            row = {k: v for k, v in r.items() if k != "text"}
            row.update({
                "match_pct": round(final * 100, 1),
                "semantic_pct": round(sem[i] * 100, 1),
                "skill_pct": round(skill_score * 100, 1),
                "experience_pct": round(exp_score * 100, 1),
                "years_experience": info["years_experience"],
                "education": info["education"],
                "email": info["email"],
                "phone": info["phone"],
                "matched_skills": ", ".join(matched),
                "missing_skills": ", ".join(missing),
            })
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("match_pct", ascending=False)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df.reset_index(drop=True)
