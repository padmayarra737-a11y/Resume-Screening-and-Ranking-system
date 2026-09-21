"""
Resume Screening & Ranking System -- Streamlit app.
Run:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.parser import parse_resume
from src.ranker import ResumeRanker, embeddings_available
from src.extractor import extract_skills

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="Resume Ranker", page_icon="📄", layout="wide")
st.title("📄 Resume Screening & Ranking System")
st.caption("Paste a job description, add resumes, and get a ranked list with match scores "
           "and the skills each candidate is missing.")


# ---------- cached data ----------
@st.cache_data
def load_dataset_resumes():
    path = DATA / "resumes_clean.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_dataset_jobs():
    path = DATA / "jobs_clean.csv"
    return pd.read_csv(path) if path.exists() else None


resume_df = load_dataset_resumes()
jobs_df = load_dataset_jobs()

# ---------- sidebar ----------
st.sidebar.header("Settings")
options = ["TF-IDF (fast)"]
if embeddings_available():
    options.append("Sentence embeddings (smarter)")
method_label = st.sidebar.radio("Similarity method", options)
method = "embeddings" if method_label.startswith("Sentence") else "tfidf"
if not embeddings_available():
    st.sidebar.caption("Embeddings not installed. Optional: `pip install sentence-transformers`")

st.sidebar.subheader("Score weights")
w_sem = st.sidebar.slider("Text similarity", 0, 100, 60, 5)
w_skill = st.sidebar.slider("Skill match", 0, 100, 30, 5)
w_exp = st.sidebar.slider("Experience match", 0, 100, 10, 5)
if w_sem + w_skill + w_exp == 0:
    st.sidebar.error("At least one weight must be above 0.")
    st.stop()
total_w = w_sem + w_skill + w_exp
weights = (w_sem / total_w, w_skill / total_w, w_exp / total_w)
st.sidebar.caption(f"Used as {weights[0]:.0%} / {weights[1]:.0%} / {weights[2]:.0%}")
st.sidebar.info("Fairness note: names, gender, photos and age are never used in scoring. "
                "Only the text content is compared with the job description.")

# ---------- 1. job description ----------
st.header("1. Job description")
sample_choices = ["(write my own)"]
if jobs_df is not None:
    sample_choices += [f"{r.job_id}: {r.title}" for r in jobs_df.head(40).itertuples()]
sample_pick = st.selectbox("Load a sample job description (optional)", sample_choices)
default_jd = ""
if sample_pick != "(write my own)" and jobs_df is not None:
    default_jd = jobs_df.loc[jobs_df.job_id == int(sample_pick.split(":")[0]), "raw_text"].iloc[0]
jd_text = st.text_area("Paste the job description here", value=default_jd, height=200, key=f"jd_{sample_pick}")
if jd_text.strip():
    st.caption("Skills detected in this job description: " + (", ".join(extract_skills(jd_text)) or "none"))

# ---------- 2. resumes ----------
st.header("2. Resumes")
source = st.radio("Where are the resumes from?",
                  ["Upload files (PDF / DOCX / TXT)", "Use resumes from the Kaggle dataset"],
                  horizontal=True)

resumes, failed = [], []
if source.startswith("Upload"):
    files = st.file_uploader("Upload one or more resumes", type=["pdf", "docx", "txt"],
                             accept_multiple_files=True)
    for f in files or []:
        try:
            text = parse_resume(f, f.name)
            if len(text.split()) < 30:
                failed.append(f"{f.name} (almost no text found - scanned PDF?)")
            else:
                resumes.append({"name": f.name, "text": text})
        except Exception as e:  # keep the app alive on a bad file
            failed.append(f"{f.name} ({e})")
else:
    if resume_df is None:
        st.warning("Run `python -m src.load_data` first to create data/resumes_clean.csv.")
    else:
        cats = sorted(resume_df["Category"].unique())
        chosen = st.multiselect("Categories to include", cats,
                                default=["INFORMATION-TECHNOLOGY", "FINANCE", "HR", "CHEF"])
        per_cat = st.slider("Resumes per category", 5, 100, 25)
        for cat in chosen:
            sub = resume_df[resume_df["Category"] == cat].head(per_cat)
            resumes += [{"name": f"{r.ID}", "category": r.Category, "text": r.raw_text}
                        for r in sub.itertuples()]
        st.caption(f"{len(resumes)} resumes selected.")

for msg in failed:
    st.warning(f"Skipped: {msg}")

# ---------- 3. rank ----------
st.header("3. Results")
if st.button("Rank resumes", type="primary"):
    if not jd_text.strip():
        st.error("Please add a job description first.")
    elif not resumes:
        st.error("Please add at least one resume.")
    else:
        background = resume_df["raw_text"].tolist() if resume_df is not None else None
        with st.spinner("Ranking..."):
            ranker = ResumeRanker(method=method, weights=weights, background_texts=background)
            st.session_state["results"] = ranker.rank(jd_text, resumes)

results = st.session_state.get("results")
if results is not None and not results.empty:
    top = results.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidates ranked", len(results))
    c2.metric("Best match", f"{top['match_pct']}%", top["name"])
    c3.metric("Average match", f"{results['match_pct'].mean():.1f}%")

    show_cols = [c for c in ["rank", "name", "category", "match_pct", "semantic_pct", "skill_pct",
                             "experience_pct", "years_experience", "education", "matched_skills",
                             "missing_skills"] if c in results.columns]
    st.dataframe(results[show_cols], width="stretch", hide_index=True)

    st.subheader("Top candidates")
    n_show = min(15, len(results))
    st.bar_chart(results.head(n_show).set_index("name")["match_pct"])

    st.subheader("Why did they rank this way?")
    for _, r in results.head(10).iterrows():
        with st.expander(f"#{r['rank']}  {r['name']}  -  {r['match_pct']}% match"):
            a, b, c = st.columns(3)
            a.metric("Text similarity", f"{r['semantic_pct']}%")
            b.metric("Skill match", f"{r['skill_pct']}%")
            c.metric("Experience match", f"{r['experience_pct']}%")
            st.write(f"**Matched skills:** {r['matched_skills'] or '-'}")
            st.write(f"**Missing skills:** {r['missing_skills'] or '-'}")
            st.write(f"**Years of experience found:** {r['years_experience'] or 'not stated'}  |  "
                     f"**Education:** {r['education']}")
            if r.get("email") or r.get("phone"):
                st.write(f"**Contact:** {r.get('email', '')} {r.get('phone', '')}")

    st.download_button("⬇️ Download ranking as CSV", results.to_csv(index=False).encode("utf-8"),
                       file_name="resume_ranking.csv", mime="text/csv")
