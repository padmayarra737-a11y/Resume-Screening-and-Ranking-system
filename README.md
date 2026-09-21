# 📄 Resume Screening & Ranking System

An NLP project that ranks resumes against a job description. A recruiter pastes a job
description, uploads resumes (PDF / DOCX / TXT), and gets a ranked list with a match score,
matched skills, **missing skills**, and a score breakdown for every candidate.

## How it works

```
Resume files ──► parser.py ──► extractor.py ──► ranker.py ──► Streamlit app
(PDF/DOCX/TXT)   raw text      skills, years,    score + rank    table, charts,
                               education,                         explanations,
                               email, phone                       CSV export
```

**Final score = 60% text similarity + 30% skill match + 10% experience match**
(weights can be changed in the app sidebar)

| Part | What it does |
|---|---|
| Text similarity | TF-IDF (1-2 word phrases) + cosine similarity. Optional sentence-transformer embeddings. |
| Skill match | % of the job's required skills found in the resume (275-skill dictionary in `data/skills.csv`) |
| Experience match | Years found in resume vs. years required by the job (capped at 100%) |

## Project structure

```
resume-ranker/
├── app.py                  # Streamlit web app
├── requirements.txt
├── data/
│   ├── raw/                # put Resume.csv and data.csv here
│   └── skills.csv          # skills dictionary (edit to add your own)
├── sample_resumes/         # 6 test resumes (PDF, DOCX, TXT)
├── sample_jds/             # sample job description
├── results/                # evaluation output (created by evaluate.py)
└── src/
    ├── load_data.py        # Step 1: clean the Kaggle datasets
    ├── parser.py           # Step 2: read PDF / DOCX / TXT
    ├── extractor.py        # Step 3: skills, experience, education, contacts
    ├── ranker.py           # Step 4: scoring and ranking
    ├── evaluate.py         # Step 5: precision@10 evaluation
    └── demo.py             # quick end-to-end check
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows      |   source venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
```

Put `Resume.csv` (Kaggle Resume Dataset) and `data.csv` (job descriptions) in `data/raw/`.

## Run it (in this order)

```bash
python -m src.load_data     # Step 1: clean data
python -m src.extractor     # Step 3: test extraction on a sample text
python -m src.demo          # Steps 2-4: rank 6 sample resumes (PDF/DOCX/TXT)
python -m src.evaluate      # Step 5: accuracy test (takes about 1-2 minutes)
streamlit run app.py        # Step 6: launch the web app
```

## Results

Test A: one hand-written job description per category, ranking all 2,481 Kaggle resumes,
**precision@10** (how many of the top 10 belong to the right category):

| Method | Average precision@10 (15 categories) |
|---|---|
| TF-IDF only | **0.71** |
| Combined (60/30/10) | 0.69 |

Best: HR, Chef, Fitness (1.00). Weakest: Information-Technology and Sales (0.10).

Test B: 30 "Data Analyst" job descriptions from `data.csv` (the resume dataset has no Data Analyst
category, so this is a sanity check, not an accuracy score). 38% of the top-10 resumes came from
analyst-type categories (IT, Finance, Business-Development, Consultant, Accountant, Banking),
and none of the top-10 lists were dominated by unrelated fields such as Chef or Fitness.

## Limitations (be honest about these in interviews)

- **Category overlap in the dataset.** Sales vs. Business-Development and IT vs.
  Engineering/Consultant resumes look very similar, which caps precision for those labels.
  The IT resumes are mostly IT support/management, not software developers.
- **Skill match did not beat TF-IDF alone** on this benchmark, but it adds explainability
  (matched / missing skills), which is what recruiters need.
- The skills list is hand-made and tech/business-heavy. Skills not in the list are invisible
  to the skill score.
- Experience is total years mentioned, not years in the relevant field.
- Scanned (image-only) PDFs contain no text and are skipped (OCR is a future improvement).
- The Kaggle resumes are labeled by category only; there is no "correct ranking" ground truth.

## Ideas to extend

- Sentence embeddings (`pip install sentence-transformers`) - selectable in the app once installed
- OCR for scanned PDFs (pytesseract)
- Learn skills automatically from job descriptions instead of a fixed list
- Deploy on Streamlit Community Cloud or Hugging Face Spaces

## Fairness

Names, gender, age and photos are never used in scoring - only text content is compared
with the job description.
