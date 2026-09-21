"""
Step 1: Load and clean the two raw datasets.

Input  : data/raw/Resume.csv   (resumes with a Category label)
         data/raw/data.csv     (job descriptions)
Output : data/resumes_clean.csv
         data/jobs_clean.csv
"""
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data"


def clean_text(text: str) -> str:
    """Lowercase, remove URLs/emails, keep letters, numbers and a few symbols
    useful for skills (c++, c#, .net), and collapse extra spaces."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)                 # urls
    text = re.sub(r"\S+@\S+", " ", text)                           # emails
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)                    # special chars
    text = re.sub(r"(?<![a-z0-9])\.|\.(?![a-z0-9])", " ", text)    # stray dots
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_resumes() -> pd.DataFrame:
    # Only read the columns we need; Resume_html is huge and not useful.
    df = pd.read_csv(RAW_DIR / "Resume.csv", usecols=["ID", "Resume_str", "Category"])
    print(f"[resumes] raw rows: {len(df)}")

    df = df.rename(columns={"Resume_str": "raw_text"})
    df["clean_text"] = df["raw_text"].apply(clean_text) 
    df["word_count"] = df["clean_text"].str.split().str.len()

    # Drop empty / very short resumes (fewer than 50 words)
    before = len(df)
    df = df[df["word_count"] >= 50]
    print(f"[resumes] removed {before - len(df)} empty/very short resumes")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset="clean_text")
    print(f"[resumes] removed {before - len(df)} duplicate resumes")

    df = df.reset_index(drop=True)
    print(f"[resumes] final rows: {len(df)}")
    return df[["ID", "Category", "raw_text", "clean_text", "word_count"]]


def load_jobs() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "data.csv")
    print(f"[jobs] raw rows: {len(df)}")

    df = df.rename(columns={"Job Title": "title", "Description": "raw_text"})
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df["word_count"] = df["clean_text"].str.split().str.len()

    # Drop very short job descriptions (fewer than 50 words)
    before = len(df)
    df = df[df["word_count"] >= 50]
    print(f"[jobs] removed {before - len(df)} very short descriptions")

    before = len(df)
    df = df.drop_duplicates(subset="clean_text")
    print(f"[jobs] removed {before - len(df)} duplicates")

    df = df.reset_index(drop=True)
    df.insert(0, "job_id", df.index)
    print(f"[jobs] final rows: {len(df)}")
    return df[["job_id", "title", "raw_text", "clean_text", "word_count"]]


def main():
    resumes = load_resumes()
    print()
    jobs = load_jobs()

    resumes.to_csv(OUT_DIR / "resumes_clean.csv", index=False)
    jobs.to_csv(OUT_DIR / "jobs_clean.csv", index=False)

    print("\nResumes per category:")
    print(resumes["Category"].value_counts().to_string())
    print("\nSaved: data/resumes_clean.csv and data/jobs_clean.csv")


if __name__ == "__main__":
    main()
