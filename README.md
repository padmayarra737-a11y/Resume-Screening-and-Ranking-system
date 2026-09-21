# 📄 Resume Screening & Ranking System

An NLP-based project that compares resumes with a job description and ranks candidates based on their relevance.

The system supports **PDF, DOCX, and TXT** resumes and provides match scores, matched skills, missing skills, and score breakdowns.

## 🚀 Features

* 📄 Upload PDF, DOCX, and TXT resumes
* 📝 Enter a job description
* 🔍 Resume text extraction
* 🤖 NLP-based resume matching
* 📊 TF-IDF + Cosine Similarity
* 🛠️ Skill matching
* ❌ Missing skill detection
* 💼 Experience matching
* 🏆 Candidate ranking
* 📈 Score breakdown
* 📥 Export results as CSV
* 🌐 Streamlit web application
## How it works
```
Resume files ──► parser.py ──► extractor.py ──► ranker.py ──► Streamlit app
(PDF/DOCX/TXT)   raw text      skills, years,    score + rank    table, charts,
                               education,                         explanations,
                               email, phone                       CSV export
```

## 🧮 Scoring

The final score is calculated as:

```text
Final Score =
0.60 × Text Similarity
+ 0.30 × Skill Match
+ 0.10 × Experience Match
```

### Text Similarity

TF-IDF with 1-2 word n-grams and cosine similarity is used to compare the resume with the job description.

### Skill Match

Required skills from the job description are compared with skills found in the resume.

Example:

```text
Matched Skills:
✓ Python
✓ SQL
✓ Machine Learning

Missing Skills:
✗ Docker
✗ AWS
```

### Experience Match

The candidate's mentioned years of experience are compared with the experience required by the job.

## 📁 Project Structure

```text
resume-screening-ranking-system/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   └── skills.csv
│
├── sample_resumes/
├── sample_jds/
├── results/
│
└── src/
    ├── load_data.py
    ├── parser.py
    ├── extractor.py
    ├── ranker.py
    ├── evaluate.py
    └── demo.py
```

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* PyPDF2
* python-docx
* Streamlit
* Matplotlib

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/padmayarra737-a11y/resume-screening-ranking-system.git
cd resume-screening-ranking-system
```

Create a virtual environment:

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```
Install dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Dataset

The project uses:

* `Resume.csv` — Resume dataset
* `data.csv` — Job description dataset
* `skills.csv` — Skills dictionary

Place the raw datasets inside:

```text
data/raw/
```

## ▶️ Run the Project

Clean the datasets:

```bash
python -m src.load_data
```

Test extraction:

```bash
python -m src.extractor
```

Run the demo:

```bash
python -m src.demo
```

Run evaluation:

```bash
python -m src.evaluate
```

Start the Streamlit application:

```bash
streamlit run app.py
```

## 📈 Evaluation

The system was tested using the resume dataset.

| Method           | Average Precision@10 |
| ---------------- | -------------------: |
| TF-IDF Only      |             **0.71** |
| Combined Scoring |                 0.69 |

The evaluation shows that TF-IDF performed well on the available category-based dataset, while the combined scoring approach provides additional **skill and experience explanations**.

## ⚠️ Limitations

* Skill matching depends on the skills dictionary.
* Experience extraction is based on years mentioned in the resume.
* Scanned/image-only PDFs are not currently supported.
* The dataset does not contain a true recruiter ranking, so Precision@10 is only an experimental evaluation.
* Resume categories can overlap, making some categories difficult to distinguish.

## 🔮 Future Improvements

* Sentence Transformer semantic matching
* OCR for scanned resumes
* Automatic skill extraction
* Better experience extraction
* Deployment using Streamlit Cloud or Hugging Face Spaces
* 
GitHub: `https://github.com/padmayarra737-a11y`
