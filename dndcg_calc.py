import os
import math
import argparse
import csv
import logging
from collections import defaultdict

# -----------------------------
# Lettura file di giudizi (qrels)
# -----------------------------
def read_qrels_file(qrels_path):
    relevance_judgments = defaultdict(dict)
    with open(qrels_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            query_id, doc_id, relevance = parts[0], parts[2], int(parts[3])
            relevance_judgments[query_id][doc_id] = relevance

    logging.info(f"Qrels loaded from {qrels_path}: {len(relevance_judgments)} query.")
    return relevance_judgments

def list_run_files(runs_directory, extensions=(".txt", ".run")):
    run_files = sorted([
        os.path.join(runs_directory, f)
        for f in os.listdir(runs_directory)
        if f.lower().endswith(extensions)
    ])

    logging.info(f"Trovati {len(run_files)} run in {runs_directory}")
    return run_files

# -----------------------------
# Metriche di ranking
# -----------------------------
def compute_dcg(relevance_grades):
    dcg = 0.0
    for rank, rel in enumerate(relevance_grades, start=1):
        dcg += rel / math.log2(rank + 1)
    return dcg

def compute_ndcg(document_list, qrels_for_query):
    if not document_list:
        return 0.0

    relevance_grades = [qrels_for_query.get(doc_id, 0) for doc_id in document_list]
    dcg = compute_dcg(relevance_grades)

    # Docs ordinati per rilevanza decrescente
    ideal_order = sorted(
        document_list,
        key=lambda d: (-qrels_for_query.get(d, 0), d)
    )
    ideal_rels = [qrels_for_query.get(d, 0) for d in ideal_order]
    idcg = compute_dcg(ideal_rels)

    return 0.0 if idcg <= 0.0 else dcg / idcg

def compute_mean_dndcg(top_docs_A, top_docs_B, qrels, k=20):
    dn_scores = []

    for query_id in set(top_docs_A) | set(top_docs_B):
        docs_A = top_docs_A.get(query_id, [])
        docs_B = set(top_docs_B.get(query_id, []))
        new_docs = [doc for doc in docs_A if doc not in docs_B]

        dndcg_value = compute_ndcg(new_docs, qrels.get(query_id, {}))
        dn_scores.append(dndcg_value)

    return (sum(dn_scores) / len(dn_scores)) if dn_scores else 0.0

# -----------------------------
# Parsing dei run TREC
# -----------------------------
def read_run_topk(path, k):
    docs_per_query = defaultdict(list)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            query_id, doc_id = parts[0], parts[2]

            if len(docs_per_query[query_id]) < k:
                docs_per_query[query_id].append(doc_id)

    logging.debug(f"[{os.path.basename(path)}] Tenuti top-{k} per {len(docs_per_query)} query.")
    return docs_per_query

# -----------------------------
# Helpers
# -----------------------------
def format_number(value: float) -> str:
    return f"{value:.6f}".replace(".", ",")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute pairwise dNDCG@k matrix for all runs in a directory."
    )
    parser.add_argument("--qrels", default="storage/queries/robust04.qrels",
                        help="Path to qrels file (TREC format).")
    parser.add_argument("--runs_dir", default="storage/runs/prf_chi_square_nmf",
                        help="Directory containing TREC runs (.txt/.run).")
    parser.add_argument("--k", type=int, default=20,
                        help="Cutoff k (default: 20).")
    parser.add_argument("--out_matrix_csv", default="storage/dndcg/pairwise_dndcg_matrix.csv",
                        help="Output CSV name.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print debug logs.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    if not os.path.isfile(args.qrels):
        raise SystemExit(f"Qrels file not found: {args.qrels}")
    if not os.path.isdir(args.runs_dir):
        raise SystemExit(f"Runs dir not found: {args.runs_dir}")

    run_files = list_run_files(args.runs_dir)
    if not run_files:
        raise SystemExit(f"No runs (.txt/.run) found in {args.runs_dir}")

    qrels = read_qrels_file(args.qrels)

    run_names, run_top_docs = [], {}
    for path in run_files:
        run_name = os.path.splitext(os.path.basename(path))[0]

        if run_name in run_names:
            base = run_name
            i = 2
            while f"{base}_{i}" in run_names:
                i += 1
            run_name = f"{base}_{i}"

        run_names.append(run_name)
        run_top_docs[run_name] = read_run_topk(path, args.k)

    logging.info(f"Computing matrix for {len(run_names)} run...")

    with open(args.out_matrix_csv, "w", newline="", encoding="utf-8") as f_mat:
        writer = csv.writer(f_mat, delimiter=";")
        writer.writerow([""] + run_names)

        for i, run_A in enumerate(run_names, 1):
            row_values = []
            logging.debug(f"Processing {run_A} ({i}/{len(run_names)})...")
            for run_B in run_names:
                val = 0.0 if run_A == run_B else compute_mean_dndcg(
                    run_top_docs[run_A], run_top_docs[run_B], qrels, args.k
                )
                row_values.append(format_number(val))
            writer.writerow([run_A] + row_values)

    logging.info(f"[OK] Wrote matrix CSV to: {args.out_matrix_csv}")


if __name__ == "__main__":
    main()
