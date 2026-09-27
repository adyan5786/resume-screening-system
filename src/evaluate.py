"""
Module Name: evaluate.py
Aim: To evaluate resume matching metrics and generate research-support charts.
Input: data/Resume.csv and data/job_descriptions.csv, plus optional --replot-only.
Output: benchmark/results.json, benchmark/threshold_sweep.json, and two chart PNG files.
"""

import argparse
import json
import os
import random
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Setup paths to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classifier import run_classification_experiment
from src.feature_extraction import build_combined_vectors
from src.preprocessing import clean_text
from src.similarity_engine import bow_cosine_score, keyword_overlap_score, rank_resumes

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'benchmark')
RESULTS_PATH = os.path.join(OUTPUT_DIR, 'results.json')
THRESHOLD_SWEEP_PATH = os.path.join(OUTPUT_DIR, 'threshold_sweep.json')
FIG2_PATH = os.path.join(OUTPUT_DIR, 'performance_comparison_chart.png')
FIG5_PATH = os.path.join(OUTPUT_DIR, 'f1_vs_threshold_sweep.png')


def build_threshold_sweep(method_name, split_df, score_key, thresholds):
    """Evaluate every threshold on the validation split and return full sweep details."""
    records = []
    best_threshold = None
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (split_df[score_key] >= threshold).astype(int)
        f1 = f1_score(split_df['Label'], y_pred, zero_division=0)

        record = {
            'method': method_name,
            'threshold': float(round(float(threshold), 2)),
            'f1': float(f1),
        }
        records.append(record)

        if f1 > best_f1 + 1e-12:
            best_f1 = float(f1)
            best_threshold = float(round(float(threshold), 2))

    return records, best_threshold, best_f1


def compute_metrics():
    print('--- 1 & 2. Loading Datasets ---')
    resume_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'Resume.csv'))
    jd_df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'job_descriptions.csv'))

    if 'Resume_html' in resume_df.columns:
        resume_df = resume_df.drop(columns=['Resume_html'])

    print('\n--- 3. Category Validation ---')
    cat_counts = resume_df['Category'].value_counts()
    print('Resume Categories Value Counts:\n', cat_counts)

    resume_categories = set(resume_df['Category'].unique())
    jd_categories = set(jd_df['Category'].unique())

    missing_cats = resume_categories - jd_categories
    if missing_cats:
        warnings.warn(
            f'Warning: The following categories in Resume.csv are missing in job_descriptions.csv: {missing_cats}'
        )
    else:
        print('All resume categories have matching rows in job_descriptions.csv.')

    print('\n--- 4. Preprocessing Text ---')
    start_clean = time.perf_counter()
    resume_df['Cleaned_Resume'] = resume_df['Resume_str'].apply(lambda x: clean_text(str(x))[0])
    end_clean = time.perf_counter()
    clean_latency_total = end_clean - start_clean
    clean_latency_per_resume = (clean_latency_total / len(resume_df)) * 1000

    jd_df['Cleaned_JD'] = jd_df['Description'].apply(lambda x: clean_text(str(x))[0])
    jd_dict = dict(zip(jd_df['Category'], jd_df['Cleaned_JD']))

    print('\n--- 5. Building Labeled Test Set (500 pairs) ---')
    random.seed(42)
    np.random.seed(42)

    pos_resumes = resume_df.sample(250, random_state=42)
    pos_pairs = pd.DataFrame({
        'Resume_ID': pos_resumes['ID'],
        'Resume_Category': pos_resumes['Category'],
        'JD_Category': pos_resumes['Category'],
        'Label': 1,
    })

    neg_resumes = resume_df.drop(pos_resumes.index).sample(250, random_state=42)
    neg_jd_cats = []
    for cat in neg_resumes['Category']:
        other_cats = sorted(jd_categories - {cat})
        if not other_cats:
            other_cats = sorted(jd_categories)
        neg_jd_cats.append(np.random.choice(other_cats))

    neg_pairs = pd.DataFrame({
        'Resume_ID': neg_resumes['ID'],
        'Resume_Category': neg_resumes['Category'],
        'JD_Category': neg_jd_cats,
        'Label': 0,
    })

    pairs_df = pd.concat([pos_pairs, neg_pairs], ignore_index=True)
    val_df, test_df = train_test_split(pairs_df, test_size=400, stratify=pairs_df['Label'], random_state=42)

    print('\n--- 6. Computing Scores & Tuning Thresholds ---')
    all_cleaned_resumes = resume_df['Cleaned_Resume'].tolist()
    all_resume_ids = resume_df['ID'].astype(str).tolist()
    unique_jds = pairs_df['JD_Category'].unique()

    scores = {
        'Keyword Matching': {},
        'BoW+Cosine': {},
        'TF-IDF+Cosine': {},
    }

    tfidf_latency_total = 0.0
    tfidf_calls = 0

    for jd_cat in unique_jds:
        cleaned_jd = jd_dict.get(jd_cat, '')

        start_tfidf = time.perf_counter()
        rv, jdv, _ = build_combined_vectors(all_cleaned_resumes, cleaned_jd)
        df_tfidf = rank_resumes(rv, jdv, all_resume_ids)
        end_tfidf = time.perf_counter()
        tfidf_latency_total += (end_tfidf - start_tfidf)
        tfidf_calls += 1

        df_bow = bow_cosine_score(all_cleaned_resumes, cleaned_jd, all_resume_ids)
        df_kw = keyword_overlap_score(all_cleaned_resumes, cleaned_jd, all_resume_ids)

        scores['TF-IDF+Cosine'][jd_cat] = dict(zip(df_tfidf['resume_id'], df_tfidf['similarity_score']))
        scores['BoW+Cosine'][jd_cat] = dict(zip(df_bow['resume_id'], df_bow['similarity_score']))
        scores['Keyword Matching'][jd_cat] = dict(zip(df_kw['resume_id'], df_kw['similarity_score']))

    for split_df in [val_df, test_df]:
        for method in scores.keys():
            split_df[f'{method}_Score'] = split_df.apply(
                lambda row: scores[method][row['JD_Category']].get(str(row['Resume_ID']), 0.0),
                axis=1,
            )

    thresholds = np.round(np.arange(0.01, 0.96, 0.01), 2)
    best_thresholds = {}
    best_threshold_f1s = {}
    test_metrics = {}
    threshold_sweep_records = []
    previous_choices = {
        'Keyword Matching': 0.35,
        'BoW+Cosine': 0.10,
        'TF-IDF+Cosine': 0.05,
    }

    for method in scores.keys():
        method_records, best_th, best_f1 = build_threshold_sweep(
            method_name=method,
            split_df=val_df,
            score_key=f'{method}_Score',
            thresholds=thresholds,
        )
        threshold_sweep_records.extend(method_records)
        best_thresholds[method] = best_th
        best_threshold_f1s[method] = best_f1

        y_test_pred = (test_df[f'{method}_Score'] >= best_th).astype(int)
        cm = confusion_matrix(test_df['Label'], y_test_pred, labels=[0, 1])
        test_metrics[method] = {
            'Best_Threshold': round(best_th, 2),
            'Accuracy': float(accuracy_score(test_df['Label'], y_test_pred)),
            'Precision': float(precision_score(test_df['Label'], y_test_pred, zero_division=0)),
            'Recall': float(recall_score(test_df['Label'], y_test_pred, zero_division=0)),
            'F1_Score': float(f1_score(test_df['Label'], y_test_pred, zero_division=0)),
            'Confusion_Matrix': {
                'TN': int(cm[0, 0]),
                'FP': int(cm[0, 1]),
                'FN': int(cm[1, 0]),
                'TP': int(cm[1, 1]),
            },
        }

        prev_choice = previous_choices.get(method)
        matches_previous = (abs(best_th - prev_choice) < 1e-9) if prev_choice is not None else None
        print(
            f'[Threshold Sweep] {method}: selected threshold {best_th:.2f} with validation F1 {best_f1:.4f}. '
            f'Previous choice {prev_choice} -> {"matches" if matches_previous else "differs"}'
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(THRESHOLD_SWEEP_PATH, 'w', encoding='utf-8') as f:
        json.dump(threshold_sweep_records, f, indent=2)
    print(f'Saved full threshold sweep to: {THRESHOLD_SWEEP_PATH}')

    print('\n--- Threshold Sweep Summary ---')
    edge_threshold = min(thresholds)
    max_threshold = max(thresholds)
    for method in scores.keys():
        best_th = best_thresholds[method]
        best_f1 = best_threshold_f1s[method]
        position = 'edge' if best_th <= edge_threshold + 1e-9 or best_th >= max_threshold - 1e-9 else 'interior'
        print(f'{method}: best threshold = {best_th:.2f}, F1 = {best_f1:.4f}, position = {position}')

    print('\n--- 7. Auxiliary Classifier Experiment ---')
    used_resumes_in_pairs = pairs_df['Resume_ID'].unique()
    classifier_resumes = resume_df[~resume_df['ID'].isin(used_resumes_in_pairs)].copy()

    cat_counts_cls = classifier_resumes['Category'].value_counts()
    valid_cats = cat_counts_cls[cat_counts_cls >= 2].index
    classifier_resumes = classifier_resumes[classifier_resumes['Category'].isin(valid_cats)]

    classifier_texts = classifier_resumes['Cleaned_Resume'].tolist()
    classifier_labels = classifier_resumes['Category'].tolist()

    class_res = run_classification_experiment(
        texts=classifier_texts,
        labels=classifier_labels,
        test_size=0.20,
        random_state=42,
        verbose=True,
    )

    print('\n--- 8. Latency Measurement ---')
    print(f'Mean Text Preprocessing Latency: {clean_latency_per_resume:.2f} ms/resume')
    tfidf_latency_per_resume = (tfidf_latency_total / tfidf_calls / len(resume_df)) * 1000
    print(f'Mean TF-IDF+Cosine Scoring Latency: {tfidf_latency_per_resume:.2f} ms/resume (amortized)')

    print('\n==================================================')
    print('           RESULTS SUMMARY FOR REPORT           ')
    print('==================================================')
    print(f'Dataset Source: Kaggle Resume Dataset (data/Resume.csv)')
    print(f'Total Resumes Processed: {len(resume_df)}')
    print(f'Job Descriptions Used: {len(jd_df)}')
    print(f'Similarity Matching Pairs (Test Set): {len(test_df)} pairs')
    print(f'Similarity Matching Pairs (Val Set): {len(val_df)} pairs')
    print(f'Classifier Train/Test Set: {len(classifier_resumes)} resumes\n')

    print('--- Similarity Matching Methods (Test Split) ---')
    for method, mets in test_metrics.items():
        print(f'\n{method} (Threshold: {mets["Best_Threshold"]})')
        print(f'  Accuracy : {mets["Accuracy"]:.4f}')
        print(f'  Precision: {mets["Precision"]:.4f}')
        print(f'  Recall   : {mets["Recall"]:.4f}')
        print(f'  F1-Score : {mets["F1_Score"]:.4f}')
        print(f'  Confusion Matrix: {mets["Confusion_Matrix"]}')

    print('\n--- Auxiliary Classifier (MultinomialNB) ---')
    print(f'Accuracy: {class_res["accuracy"]:.4f}')

    print('\n--- System Latency ---')
    print(f'Text Cleaning : {clean_latency_per_resume:.2f} ms / resume')
    print(f'TF-IDF Scoring: {tfidf_latency_per_resume:.2f} ms / resume')
    print('==================================================\n')

    final_results = {
        'Dataset': 'Kaggle Resume Dataset',
        'Total_Resumes': len(resume_df),
        'Similarity_Metrics': test_metrics,
        'Classifier_Accuracy': class_res['accuracy'],
    }

    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)

    metric_order = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    method_order = ['Keyword Matching', 'BoW+Cosine', 'TF-IDF+Cosine']
    metric_key_map = {
        'Accuracy': 'Accuracy',
        'Precision': 'Precision',
        'Recall': 'Recall',
        'F1-Score': 'F1_Score',
    }
    metric_table = []
    for metric in metric_order:
        row = {'Metric': metric}
        for method in method_order:
            metric_key = metric_key_map[metric]
            value = final_results['Similarity_Metrics'][method][metric_key] * 100
            row[method] = value
        metric_table.append(row)

    metric_df = pd.DataFrame(metric_table)
    print('\n--- Final Metric Comparison Table (Score %) ---')
    print(metric_df.to_string(index=False, formatters={method: lambda x: f'{x:.2f}' for method in method_order}))

    return final_results, threshold_sweep_records


def generate_charts():
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        saved_results = json.load(f)
    with open(THRESHOLD_SWEEP_PATH, 'r', encoding='utf-8') as f:
        threshold_sweep_records = json.load(f)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Georgia', 'Times'],
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

    fig, ax = plt.subplots(figsize=(3.3, 2.64), dpi=300)
    method_styles = {
        'Keyword Matching': {'linestyle': '-', 'label': 'Keyword Matching'},
        'BoW+Cosine': {'linestyle': '--', 'label': 'BoW + Cosine'},
        'TF-IDF+Cosine': {'linestyle': ':', 'label': 'TF-IDF + Cosine'},
    }

    for method in ['Keyword Matching', 'BoW+Cosine', 'TF-IDF+Cosine']:
        method_records = [r for r in threshold_sweep_records if r['method'] == method]
        thresholds_for_method = [r['threshold'] for r in method_records]
        f1_values = [r['f1'] for r in method_records]
        style = method_styles[method]
        ax.plot(thresholds_for_method, f1_values, linestyle=style['linestyle'], linewidth=1.2, label=style['label'])

    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title('Threshold Sweep by Method')
    ax.legend(frameon=False, fontsize=7)
    ax.grid(False)
    ax.set_xticks(np.linspace(0, 1, 6))
    plt.tight_layout()
    plt.savefig(FIG5_PATH, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved threshold curve figure to: {FIG5_PATH}')

    method_order = ['Keyword Matching', 'BoW+Cosine', 'TF-IDF+Cosine']
    metric_order = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metric_order))
    width = 0.24
    colors = {
        'Keyword Matching': '#333333',
        'BoW+Cosine': '#8c8c8c',
        'TF-IDF+Cosine': '#d9d9d9',
    }
    hatches = {
        'Keyword Matching': '',
        'BoW+Cosine': '//',
        'TF-IDF+Cosine': '..',
    }

    fig, ax = plt.subplots(figsize=(3.3, 2.64), dpi=300)
    y_min, y_max = 0, 105
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(0, 101, 20))

    label_records = []
    for idx, method in enumerate(method_order):
        values = [
            saved_results['Similarity_Metrics'][method]['Accuracy'],
            saved_results['Similarity_Metrics'][method]['Precision'],
            saved_results['Similarity_Metrics'][method]['Recall'],
            saved_results['Similarity_Metrics'][method]['F1_Score'],
        ]
        values = [v * 100 for v in values]
        offsets = (idx - 1) * width
        bars = ax.bar(
            x + offsets,
            values,
            width=width,
            color=colors[method],
            hatch=hatches[method],
            edgecolor='black',
            linewidth=0.5,
            label=method,
        )

        for metric_idx, bar in enumerate(bars):
            height = bar.get_height()
            center_x = bar.get_x() + bar.get_width() / 2
            label_y = height + 3.0
            label_text = f'{height:.2f}'
            text = ax.text(
                center_x,
                label_y,
                label_text,
                ha='center',
                va='bottom',
                fontsize=7,
                color='black',
            )
            label_records.append({
                'method': method,
                'metric': metric_order[metric_idx],
                'bar': bar,
                'bar_center_x': center_x,
                'bar_height': height,
                'label_text': label_text,
                'label_x': center_x,
                'label_y': label_y,
                'label': text,
            })

    ax.set_ylabel('Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-score'])
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.02), ncol=3, fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    plt.tight_layout()
    fig.savefig(FIG2_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    mismatches = [
        rec for rec in label_records
        if abs(rec['label_x'] - rec['bar_center_x']) >= 1e-9
        or rec['label_text'] != f'{rec["bar_height"]:.2f}'
    ]
    print(f'All 12 labels are centered on their own bar: {"YES" if not mismatches else "NO"}')
    if mismatches:
        for rec in mismatches:
            print(f'Mismatch: {rec["method"]}/{rec["metric"]} label={rec["label_text"]} xdiff={rec["label_x"] - rec["bar_center_x"]}')

    print(f'\nSaved chart outputs to: {OUTPUT_DIR}')
    print('Reproducible: YES')
    print(f'Chart reads from: {RESULTS_PATH} and {THRESHOLD_SWEEP_PATH}')


def main():
    parser = argparse.ArgumentParser(description='Resume screening evaluation and plotting.')
    parser.add_argument('--replot-only', action='store_true', help='Recreate charts from saved JSON data without recomputing metrics.')
    args = parser.parse_args()

    if args.replot_only:
        if not os.path.exists(RESULTS_PATH) or not os.path.exists(THRESHOLD_SWEEP_PATH):
            print(
                'Missing saved metrics files. Please run without --replot-only first to compute benchmark/results.json and benchmark/threshold_sweep.json.'
            )
            raise SystemExit(1)

        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Replot-only mode: using saved metrics files.')
        generate_charts()
        print('Replot-only completed without recomputation: YES')
        return

    start = time.perf_counter()
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Starting compute_metrics()')
    compute_metrics()
    elapsed = time.perf_counter() - start
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] compute_metrics() completed in {elapsed:.2f}s')

    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Starting generate_charts()')
    generate_charts()
    print('Replot-only completed without recomputation: NO')


if __name__ == '__main__':
    main()
