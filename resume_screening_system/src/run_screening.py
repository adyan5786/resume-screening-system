"""
Module Name: run_screening.py
Aim: Rank raw resumes against a built-in or custom job description.
Input: --resumes_dir plus exactly one of --jd_category or --jd_file.
Output: Percentage-formatted ranked_resumes.csv in output/.
"""

import argparse
import os
import sys
import pandas as pd

# Add the project root to sys.path so 'src' can be imported easily
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import extraction
from src import preprocessing
from src.feature_extraction import build_combined_vectors
from src.similarity_engine import rank_resumes


def load_job_descriptions_csv():
    """Load the canonical job description catalog from the project data folder."""
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'job_descriptions.csv'))
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Job descriptions catalog not found at '{csv_path}'.")
    return pd.read_csv(csv_path)


def resolve_job_description(jd_category=None, jd_file=None):
    """Resolve either a category lookup or a custom job-description file."""
    provided_inputs = [bool(jd_category), bool(jd_file)]
    if sum(provided_inputs) != 1:
        raise ValueError(
            "Exactly one of --jd_category or --jd_file must be provided. "
            "Use --jd_category to look up a built-in job description, or --jd_file for a custom text file."
        )

    if jd_category is not None:
        jd_df = load_job_descriptions_csv()
        normalized = jd_df['Category'].astype(str).str.strip()
        match = normalized.str.lower() == str(jd_category).strip().lower()
        if not match.any():
            valid_categories = sorted(normalized.unique().tolist())
            print("Error: No job description found for category '{}'".format(jd_category))
            print("Valid category names:")
            for category in valid_categories:
                print(f"  - {category}")
            raise ValueError(f"Unknown job category: {jd_category}")
        row = jd_df.loc[match].iloc[0]
        return row['Description']

    if not os.path.exists(jd_file):
        raise FileNotFoundError(f"Job description file '{jd_file}' not found.")

    try:
        with open(jd_file, 'r', encoding='utf-8') as f:
            jd_text = f.read()
    except Exception as exc:
        raise ValueError(f"Failed to read JD file: {exc}") from exc

    if not jd_text.strip():
        raise ValueError("Error: Job description is empty.")

    return jd_text


def main():
    parser = argparse.ArgumentParser(description="Run the automated resume screening system.")
    parser.add_argument('--resumes_dir', type=str, required=True,
                        help="Path to the directory containing resumes (PDF/DOCX/TXT).")
    parser.add_argument('--jd_category', type=str, default=None,
                        help="Built-in job category name to look up in data/job_descriptions.csv.")
    parser.add_argument('--jd_file', type=str, default=None,
                        help="Path to a custom job description text file.")

    args = parser.parse_args()

    resumes_dir = args.resumes_dir

    try:
        jd_text = resolve_job_description(jd_category=args.jd_category, jd_file=args.jd_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if not os.path.exists(resumes_dir):
        print(f"Error: Resumes directory '{resumes_dir}' not found.")
        sys.exit(1)

    source_label = f"category '{args.jd_category}'" if args.jd_category is not None else f"file '{args.jd_file}'"
    print(f"Loading Job Description from {source_label}...")

    print("Extracting text from resumes...")
    resume_filenames = []
    resume_texts = []

    valid_extensions = {'.pdf', '.docx', '.txt'}
    
    for filename in os.listdir(resumes_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            file_path = os.path.join(resumes_dir, filename)
            try:
                extracted = extraction.extract_text(file_path)
                if extracted.strip():
                    resume_filenames.append(filename)
                    resume_texts.append(extracted)
                else:
                    print(f"Warning: Extracted text is empty for {filename}. Skipping.")
            except Exception as e:
                print(f"Warning: Failed to extract text from {filename} ({e}). Skipping.")

    if not resume_texts:
        print("Error: No valid resumes found/extracted in the provided directory.")
        sys.exit(1)

    print(f"Extracted {len(resume_texts)} resumes successfully.")

    print("Cleaning Job Description...")
    cleaned_jd = preprocessing.clean_text(jd_text)[0]

    print("Cleaning Resumes...")
    cleaned_resumes = []
    for text in resume_texts:
        cleaned_resumes.append(preprocessing.clean_text(text)[0])

    print("Vectorizing and computing similarities (TF-IDF + Cosine)...")
    # feature_extraction.build_combined_vectors handles fitting on resumes + jd
    resume_vectors, jd_vector, _ = build_combined_vectors(cleaned_resumes, cleaned_jd)

    # similarity_engine.rank_resumes computes cosine similarity and returns a DataFrame
    ranked_df = rank_resumes(resume_vectors, jd_vector, resume_ids=resume_filenames)
    
    # Rename 'resume_id' column to 'filename' and format percentages as user-facing strings.
    ranked_df = ranked_df.rename(columns={'resume_id': 'filename', 'similarity_score': 'match_percent'})
    ranked_df['match_percent'] = (ranked_df['match_percent'] * 100).map(lambda value: f"{value:.1f}%")

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'ranked_resumes.csv')

    print("\n--- Ranked Candidates ---")
    print(ranked_df.to_string(index=False))

    try:
        ranked_df.to_csv(output_file, index=False)
        print(f"\nResults successfully saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")

if __name__ == "__main__":
    main()
