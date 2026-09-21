"""
Step 3: Pull structured information out of resume / job-description text.

Extracts: skills, years of experience, education level, email, phone.
"""
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.load_data import clean_text

SKILLS_PATH = Path(__file__).resolve().parent.parent / "data" / "skills.csv"

EDU_LEVELS = [
    # (label, rank, regex) -- checked from highest to lowest
    ("PhD", 4, r"\b(ph\.?\s?d|doctorate|doctoral)\b"),
    ("Master's", 3, r"\b(master'?s?|m\.?sc|m\.?s\.?\b|mba|m\.?tech|m\.?a\.?\b|m\.?e\.?\b)"),
    ("Bachelor's", 2, r"\b(bachelor'?s?|b\.?sc|b\.?s\.?\b|b\.?tech|b\.?e\.?\b|b\.?a\.?\b|b\.?com|bba|undergraduate)"),
    ("Associate/Diploma", 1, r"\b(associate'?s?|diploma|certificate)\b"),
]


@lru_cache(maxsize=1)
def load_skill_lookup():
    """
    Build {phrase -> skill} for every skill name and alias, normalised with the
    same clean_text() used on resumes. Also returns the longest phrase length
    (in words) so we know how many words to look at together.
    """
    df = pd.read_csv(SKILLS_PATH).fillna("")
    lookup, max_n = {}, 1
    for _, row in df.iterrows():
        names = [row["skill"]] + [a for a in str(row["aliases"]).split("|") if a]
        for name in names:
            phrase = clean_text(name)
            lookup[phrase] = row["skill"]
            max_n = max(max_n, len(phrase.split()))
    return lookup, max_n


def extract_skills(text: str) -> list[str]:
    """Slide a 1..N word window over the text and look each phrase up in the
    skills dictionary (one fast pass, no regex per skill)."""
    lookup, max_n = load_skill_lookup()
    tokens = clean_text(text).split()
    found = set()
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            skill = lookup.get(" ".join(tokens[i:i + n]))
            if skill:
                found.add(skill)
    return sorted(found)


def skill_categories() -> dict:
    df = pd.read_csv(SKILLS_PATH).fillna("")
    return dict(zip(df["skill"], df["category"]))


def extract_years_experience(text: str) -> int:
    """Highest 'N years' mention (capped at 40). 0 if none found."""
    matches = re.findall(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b", text.lower())
    years = [int(m) for m in matches if 0 < int(m) <= 40]
    return max(years) if years else 0


def extract_required_years(jd_text: str) -> int:
    """Smallest 'N years' requirement in a job description (0 if none)."""
    matches = re.findall(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b", jd_text.lower())
    years = [int(m) for m in matches if 0 < int(m) <= 15]
    return min(years) if years else 0


def extract_education(text: str) -> str:
    low = text.lower()
    for label, _, pattern in EDU_LEVELS:
        if re.search(pattern, low):
            return label
    return "Not found"


def extract_email(text: str) -> str:
    m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    return m.group(0) if m else ""


def extract_phone(text: str) -> str:
    m = re.search(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}", text)
    return m.group(0).strip() if m else ""


def extract_all(text: str) -> dict:
    return {
        "skills": extract_skills(text),
        "years_experience": extract_years_experience(text),
        "education": extract_education(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
    }


if __name__ == "__main__":
    sample = (
        "Jane Doe | jane@mail.com | +1 415 555 1234\n"
        "Data Analyst with 5+ years of experience in Python, SQL, Power BI and Excel. "
        "Master of Science in Statistics. Built ETL pipelines on AWS."
    )
    for k, v in extract_all(sample).items():
        print(f"{k:17}: {v}")
