# Resume Screening System

An NLP-based system that automatically matches candidate resumes against a
job description using TF-IDF vectorization and cosine similarity, so a
recruiter can rank a batch of applicants by relevance instead of reading
every resume manually.

Built as an academic project (B.E. Computer Engineering, Mumbai University
Rev. 2019 syllabus).

## How it works

1. **Text extraction** — pulls raw text out of resume PDF/DOCX files.
2. **Preprocessing** — lowercases, strips punctuation/digits, tokenizes,
   removes stopwords, and lemmatizes both resumes and job descriptions.
3. **Feature extraction** — converts cleaned text into TF-IDF vectors,
   fit on the combined corpus so resumes and job descriptions share the
   same vocabulary space.
4. **Similarity scoring** — computes cosine similarity between each resume
   vector and the job description vector.
5. **Ranking** — sorts resumes by similarity score, highest first.

## Two ways to use it

| Script | Question it answers | Who it's for |
|---|---|---|
| `run_screening.py` | "I have one open role — who's the best fit among these candidates?" | Recruiter / HR — the system's primary purpose |
| `match_categories.py` | "I have this resume — which of many job types does it best fit?" | Candidate self-check, or validation/testing tool |

## Real measured results

Benchmarked on the Kaggle "Resume Dataset" (2,484 resumes, 24 job
categories) against 500 labeled resume–job description pairs (400 held-out
test pairs), comparing three matching methods:

| Method | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Keyword Matching | 68.00% | 68.95% | 65.50% | 67.18% |
| Bag-of-Words + Cosine | 74.00% | 72.64% | 77.00% | 74.76% |
| **TF-IDF + Cosine (proposed)** | **82.00%** | **86.78%** | 75.50% | **80.75%** |

Latency: ~4.4 ms/resume for preprocessing, ~1.4 ms/resume for TF-IDF +
cosine scoring — screening hundreds of resumes takes a few seconds.

Full methodology, threshold-tuning details, and discussion are in the
project report and IEEE paper (not included in this repo / add your own
link here if you host them elsewhere).

## Project structure

```
resume_screening_system/
├── benchmark/              # Research/report outputs (regenerate via evaluate.py)
│   ├── performance_comparison_chart.png
│   ├── f1_vs_threshold_sweep.png
│   ├── results.json
│   └── threshold_sweep.json
├── data/
│   ├── raw_resumes/        # Sample PDFs for extraction/screening demos
│   ├── Resume.csv          # Kaggle "Resume Dataset" (Resume_str, Category)
│   └── job_descriptions.csv # One authored job description per category
├── output/                 # Core product outputs
│   ├── ranked_resumes.csv
│   ├── best_category_matches.csv
│   └── category_match_matrix.csv
├── src/
│   ├── extraction.py       # PDF/DOCX text extraction
│   ├── preprocessing.py    # Text cleaning pipeline
│   ├── feature_extraction.py # TF-IDF vectorization
│   ├── similarity_engine.py  # Cosine similarity + ranking
│   ├── classifier.py       # Auxiliary Naive Bayes category classifier
│   ├── run_screening.py    # Recruiter mode: one JD -> ranked resumes
│   ├── match_categories.py # Candidate mode: one resume -> best-fit category
│   └── evaluate.py         # Full benchmark (produces benchmark/ outputs)
├── tests/                  # Unit tests
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

## Usage

**Recruiter mode** — rank resumes against a specific job:
```bash
# Use one of the 24 built-in job categories
python src/run_screening.py --resumes_dir data/raw_resumes --jd_category "INFORMATION-TECHNOLOGY"

# Or supply your own custom job description
python src/run_screening.py --resumes_dir data/raw_resumes --jd_file my_job_posting.txt
```
Output: `output/ranked_resumes.csv` — filename, match percentage, sorted
descending.

**Candidate/validation mode** — find the best-fitting category for each resume:
```bash
python src/match_categories.py --resumes_dir data/raw_resumes
```
Output: `output/best_category_matches.csv` (top match + top 3 per resume) and
`output/category_match_matrix.csv` (full resume × category similarity
matrix).

**Run the full benchmark** (only needed to reproduce the results table above):
```bash
python src/evaluate.py              # full run: recomputes everything
python src/evaluate.py --replot-only  # fast: regenerates figures from saved results.json only
```

## Known limitations

- Lexical overlap only — cannot recognize synonyms or paraphrased skills
  (e.g. "developer" vs. "programmer").
- No deep contextual understanding, unlike transformer-based models
  (BERT, Sentence-BERT).
- Accuracy depends on resume text quality; unusual PDF layouts or scanned
  images can degrade text extraction.
- Cannot assess soft skills, cultural fit, or anything outside the resume
  text itself.
- Ground-truth relevance labels for benchmarking are proxied by job-category
  match, not human judgment.

## Future scope

- Semantic embeddings (Sentence-BERT) to catch paraphrased/synonymous skills.
- Named Entity Recognition to more precisely separate skills, degrees, and
  experience.
- A web-based UI (e.g. Streamlit) for non-technical recruiter use.
- Integration with existing Applicant Tracking Systems (ATS).

## Note on included data

`data/raw_resumes/` in this repository contains a small set of sample
resumes used for demonstrating the extraction and ranking pipeline. If you
add your own or others' resumes to this folder, be mindful of the personal
information they contain before committing or sharing.