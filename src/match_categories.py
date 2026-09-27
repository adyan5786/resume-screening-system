"""
Module Name: match_categories.py
Aim: Find the best-fitting job category for each raw resume.
Input: --resumes_dir and data/job_descriptions.csv.
Output: Percentage-formatted category_match_matrix.csv and best_category_matches.csv in output/.
"""

import argparse
import os
import sys
import pandas as pd

# Add the project root to sys.path so 'src' can be imported easily
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import extraction
from src import preprocessing
from src.feature_extraction import fit_tfidf_vectorizer, transform_tfidf
from src.similarity_engine import rank_resumes

def main():
    parser = argparse.ArgumentParser(description="Find best-fitting job category for each resume.")
    parser.add_argument('--resumes_dir', type=str, required=True, 
                        help="Path to the directory containing raw resumes (PDF/DOCX/TXT).")
    
    args = parser.parse_args()
    resumes_dir = args.resumes_dir
    
    # Assuming job descriptions are statically placed here for the purpose of category matching
    jd_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'job_descriptions.csv')

    if not os.path.exists(resumes_dir):
        print(f"Error: Resumes directory '{resumes_dir}' not found.")
        sys.exit(1)

    if not os.path.exists(jd_file):
        print(f"Error: Job descriptions file '{jd_file}' not found.")
        sys.exit(1)

    print("1. Loading Job Descriptions...")
    try:
        jd_df = pd.read_csv(jd_file)
    except Exception as e:
        print(f"Error reading job descriptions file: {e}")
        sys.exit(1)

    if 'Category' not in jd_df.columns or 'Description' not in jd_df.columns:
        print("Error: job_descriptions.csv must contain 'Category' and 'Description' columns.")
        sys.exit(1)

    jd_categories = jd_df['Category'].tolist()
    jd_texts = jd_df['Description'].tolist()
    
    print(f"   Loaded {len(jd_categories)} distinct job categories.")

    print("\n2. Extracting text from resumes...")
    resume_filenames = []
    resume_texts = []
    valid_extensions = {'.pdf', '.docx', '.txt'}

    for filename in os.listdir(resumes_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            filepath = os.path.join(resumes_dir, filename)
            try:
                extracted = extraction.extract_text(filepath)
                if extracted.strip():
                    resume_filenames.append(filename)
                    resume_texts.append(extracted)
                else:
                    print(f"Warning: Extracted text is empty for {filename}. Skipping.")
            except Exception as e:
                print(f"Warning: Failed to extract {filename} ({e}). Skipping.")

    if not resume_texts:
        print("Error: No valid resumes found/extracted in the provided directory.")
        sys.exit(1)

    print(f"   Extracted {len(resume_texts)} resumes successfully.")

    print("\n3. Cleaning text (Resumes and JDs)...")
    cleaned_resumes = [preprocessing.clean_text(t)[0] for t in resume_texts]
    cleaned_jds = [preprocessing.clean_text(t)[0] for t in jd_texts]

    print("\n4. Fitting TF-IDF Vectorizer on combined corpus...")
    # Fit ONE vectorizer on the combined corpus to ensure a shared feature space
    combined_corpus = cleaned_resumes + cleaned_jds
    vectorizer = fit_tfidf_vectorizer(combined_corpus)
    
    # Transform both independently
    resume_vectors = transform_tfidf(vectorizer, cleaned_resumes)
    jd_vectors = transform_tfidf(vectorizer, cleaned_jds)

    print("\n5. Computing cosine similarities...")
    # Reuse the shared similarity engine for each category while preserving the matrix output.
    sim_columns = []
    for category, jd_vector in zip(jd_categories, jd_vectors):
        category_scores = rank_resumes(resume_vectors, jd_vector, resume_filenames)
        sim_columns.append(category_scores.set_index('resume_id')['similarity_score'].rename(category))

    sim_matrix = pd.concat(sim_columns, axis=1).reindex(resume_filenames)

    matrix_df = pd.DataFrame(sim_matrix, index=resume_filenames, columns=jd_categories)
    
    # Convert all similarity values in matrix to percentages
    matrix_df = (matrix_df * 100).round(1)

    # Save the full matrix
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    os.makedirs(output_dir, exist_ok=True)
    matrix_csv_path = os.path.join(output_dir, 'category_match_matrix.csv')
    matrix_df.to_csv(matrix_csv_path)

    print("\n6. Finding best matches...")
    summary_data = []

    for idx, filename in enumerate(resume_filenames):
        # Scores are now percentages
        scores = matrix_df.iloc[idx]
        sorted_scores = scores.sort_values(ascending=False)
        
        best_cat = sorted_scores.index[0]
        best_score_pct = sorted_scores.iloc[0]
        second_best_score_pct = sorted_scores.iloc[1] if len(sorted_scores) > 1 else 0.0
        
        # Confidence gap: difference of > 5.0 (which is 0.05 * 100)
        confidence = "Strong match" if (best_score_pct - second_best_score_pct) > 5.0 else "Weak/ambiguous match"
        
        # Grab top 3 categories and format them
        top3_cats = sorted_scores.index[:3].tolist()
        top3_scores_pct = sorted_scores.iloc[:3].tolist()
        top3_str = ", ".join([f"{c} ({s:.1f}%)" for c, s in zip(top3_cats, top3_scores_pct)])
        
        summary_data.append({
            'filename': filename,
            'best_matching_category': best_cat,
            'match_percent': f"{best_score_pct:.1f}%",
            'confidence': confidence,
            'top_3_categories': top3_str
        })

    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_dir, 'best_category_matches.csv')
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n==================================================")
    print("             BEST CATEGORY MATCHES              ")
    print("==================================================")
    print(summary_df.to_string(index=False))

    print(f"\nSaved full N x M similarity matrix to: {matrix_csv_path}")
    print(f"Saved top-3 category summaries to: {summary_csv_path}")

if __name__ == "__main__":
    main()
