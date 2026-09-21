"""
Step 5: Measure how good the ranking is.

Test A (accuracy): one hand-written job description per resume category.
  For each JD we rank all resumes and check how many of the top 10 really belong
  to that category  ->  precision@10.

Test B (demo): rank all resumes against the Data Analyst job descriptions from
  data.csv and show which categories come out on top.

Run:  python -m src.evaluate
"""
import pandas as pd

from src.ranker import ResumeRanker
from src.load_data import OUT_DIR

BASE = OUT_DIR.parent
TEST_JDS = {
    "INFORMATION-TECHNOLOGY": "We are hiring a software engineer / IT specialist with 3+ years of experience in Java, Python, SQL, and web development. You will build applications, manage servers and networks, troubleshoot technical support issues, work with AWS cloud, Linux, and databases, and follow agile practices.",
    "ACCOUNTANT": "Seeking an accountant with 3+ years of experience in general ledger, accounts payable, accounts receivable, bank reconciliation, month-end close, financial statements, GAAP and tax preparation. Proficiency in QuickBooks and Excel required. CPA preferred.",
    "HR": "Human resources generalist needed for recruiting, onboarding, employee relations, benefits administration, payroll, performance management and HRIS. Must know employment law and support managers and employees across the company.",
    "CHEF": "Experienced chef needed to run our kitchen. Responsibilities include menu planning, food preparation, cooking, food safety and sanitation, inventory ordering, catering events, and training and supervising line cooks and kitchen staff.",
    "TEACHER": "Classroom teacher wanted to plan lessons, teach students, manage classroom behavior, assess student progress, communicate with parents, and develop curriculum. Teaching certification and experience with elementary or secondary education preferred.",
    "HEALTHCARE": "Registered nurse or clinical staff needed for patient care, medication administration, vital signs, medical records, working with physicians, infection control, HIPAA compliance, and providing care in a hospital or clinic setting.",
    "FITNESS": "Fitness trainer needed to lead personal training and group fitness classes, design workout programs, coach members on strength training and nutrition, maintain gym equipment, and hold CPR and personal trainer certification.",
    "ENGINEERING": "Mechanical engineer with 5+ years of experience in product design, CAD, SolidWorks, manufacturing processes, quality control, testing, technical drawings, and cross-functional project work in an industrial setting.",
    "DESIGNER": "Creative graphic designer wanted, skilled in Adobe Photoshop, Illustrator, InDesign, layout, typography, branding, web and print design, and working with clients and marketing teams to deliver visual concepts.",
    "SALES": "Sales representative needed to generate leads, prospect and close new customers, manage a sales pipeline, meet quarterly sales targets, deliver product demos, and maintain client relationships using CRM software.",
    "BANKING": "Bank teller / banking associate to handle customer deposits and withdrawals, loans, account opening, balancing cash drawers, complying with banking regulations, and cross-selling financial products to customers at a branch.",
    "AVIATION": "Aviation professional needed for aircraft operations: flight crew, pilot or aircraft maintenance technician with FAA certifications, flight safety procedures, airline operations, pre-flight inspections and logbook records.",
    "CONSTRUCTION": "Construction project manager / supervisor to oversee building projects, subcontractors, blueprints, schedules, budgets, OSHA site safety, permits, concrete, framing and quality inspections on residential and commercial sites.",
    "ADVOCATE": "Advocate / attorney or legal advocate to represent clients, conduct legal research, draft motions and contracts, appear in court, advise on legal rights, and support victims and community members with case management.",
    "PUBLIC-RELATIONS": "Public relations specialist to write press releases, manage media relations, build the company brand, run social media, organize events, handle crisis communications and pitch stories to journalists.",
}
DATA_ANALYST_GROUP = ("INFORMATION-TECHNOLOGY", "FINANCE", "BUSINESS-DEVELOPMENT",
                      "CONSULTANT", "ACCOUNTANT", "BANKING")


def load_resumes():
    df = pd.read_csv(OUT_DIR / "resumes_clean.csv")
    return df, [{"name": str(r.ID), "category": r.Category, "text": r.raw_text}
                for r in df.itertuples()]


def precision_at_k(ranked: pd.DataFrame, category: str, k=10) -> float:
    return (ranked.head(k)["category"] == category).mean()


def test_a(resumes, background):
    configs = {
        "TF-IDF only": ResumeRanker(weights=(1, 0, 0), background_texts=background),
        "Combined (60/30/10)": ResumeRanker(weights=(0.6, 0.3, 0.1), background_texts=background),
    }
    rows = []
    for cat, jd in TEST_JDS.items():
        row = {"category": cat}
        for name, ranker in configs.items():
            row[name] = precision_at_k(ranker.rank(jd, resumes), cat)
        rows.append(row)
    res = pd.DataFrame(rows)
    mean = res.drop(columns="category").mean()
    res.loc[len(res)] = ["AVERAGE"] + mean.tolist()
    return res


def test_b(resumes, background, n_jobs=30):
    jobs = pd.read_csv(OUT_DIR / "jobs_clean.csv")
    jobs = jobs[jobs["title"].str.lower().str.contains("data analyst")].head(n_jobs)
    ranker = ResumeRanker(background_texts=background)
    cats = []
    for jd in jobs["raw_text"]:
        cats += ranker.rank(jd, resumes).head(10)["category"].tolist()
    dist = pd.Series(cats).value_counts(normalize=True).mul(100).round(1)
    return len(jobs), dist


if __name__ == "__main__":
    df, resumes = load_resumes()
    background = [r["text"] for r in resumes]

    print("=== Test A: precision@10 with one JD per category ===")
    res = test_a(resumes, background)
    print((res.set_index("category") * 1).round(2).to_string())
    (BASE / "results").mkdir(exist_ok=True)
    res.to_csv(BASE / "results" / "evaluation_precision_at_10.csv", index=False)

    print("\n=== Test B: top-10 categories for Data Analyst job descriptions ===")
    n, dist = test_b(resumes, background)
    print(f"({n} Data Analyst JDs from data.csv, top 10 resumes each, share in %)")
    print(dist.head(10).to_string())
    share = dist[dist.index.isin(DATA_ANALYST_GROUP)].sum()
    print(f"\nShare from analyst-type categories {DATA_ANALYST_GROUP}: {share:.1f}%")
    dist.to_csv(BASE / "results" / "data_analyst_top10_categories.csv", header=["percent"])
    print("\nSaved results to the results/ folder.")
