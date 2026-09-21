"""
Quick check that everything works: rank the 6 sample resumes (PDF/DOCX/TXT)
against sample_jds/data_analyst.txt.

Run:  python -m src.demo
"""
from pathlib import Path

import pandas as pd

from src.parser import parse_resume
from src.ranker import ResumeRanker

BASE = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    jd = (BASE / "sample_jds" / "data_analyst.txt").read_text()
    resumes = [{"name": p.name, "text": parse_resume(p)}
               for p in sorted((BASE / "sample_resumes").glob("*"))]
    print(f"Parsed {len(resumes)} resumes\n")

    # Learn word importance from the Kaggle resumes if available (better scores)
    csv = BASE / "data" / "resumes_clean.csv"
    background = pd.read_csv(csv)["raw_text"].tolist() if csv.exists() else None
    if background:
        print(f"Word importance learned from {len(background)} Kaggle resumes.\n")
    else:
        print("NOTE: data/resumes_clean.csv not found - run `python -m src.load_data` first "
              "for more reliable scores.\n")

    out = ResumeRanker(background_texts=background).rank(jd, resumes)
    pd.set_option("display.width", 200)
    print(out[["rank", "name", "match_pct", "semantic_pct", "skill_pct",
               "experience_pct", "years_experience"]].to_string(index=False))
    print("\nTop candidate details")
    top = out.iloc[0]
    print("  matched skills:", top["matched_skills"])
    print("  missing skills:", top["missing_skills"])
