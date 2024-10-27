from parsers import QueryParser, QrelsParser
from sklearn.metrics import precision_score, recall_score, average_precision_score

import argparse
import es_helpers
import os
import nltk
import pandas as pd
import random
import sys

def setup():
    random.seed(a=754)
    nltk.download('stopwords', quiet=True)

    es = es_helpers.connect_elasticsearch()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--index", action="store_true", help="Index files")
    parser.add_argument("-d", "--delete-index", action="store_true", help="Delete index before indexing")
    parser.add_argument("-e", "--evaluate", action="store_true", help="Evaluate search engine performance")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print step by step results")

    args = parser.parse_args()

    return es, args

def get_leaf_directories(directory):
    leaf_directories = []

    for root, dirs, files in os.walk(directory):
        if not dirs:
            leaf_directories.append(root)

    return leaf_directories

def create_output_directory(original_directory, base_output_dir):
    relative_path = os.path.relpath(original_directory, start='tipster')
    output_dir = os.path.join(base_output_dir, relative_path)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir

def create_document_term_matrix(term_vectors):
    unique_terms = set()
    for id, terms in term_vectors.items():
        unique_terms.update(terms.keys())
    
    unique_terms = sorted(unique_terms)
    
    matrix = []
    for id, terms in term_vectors.items():
        row = []
        for term in unique_terms:
            row.append(terms.get(term, {}).get('term_freq', 0))
        matrix.append(row)

    document_term_matrix = pd.DataFrame(matrix, columns=unique_terms, index=term_vectors.keys())

    return document_term_matrix

def evaluate(es, index):
    precision_list = []
    recall_list = []
    ap_list = []

    query_parser = QueryParser('storage/queries/robust04.topics')
    qrels_parser = QrelsParser('storage/queries/robust04.qrels')
    
    queries = query_parser.parse_queries()
    qrels = qrels_parser.parse_qrels()
    
    for query_id, query in queries.items():
        search_results = es_helpers.search(es, index, query)
        
        retrieved_docs = [hit['_id'] for hit in search_results['hits']['hits']]
        
        relevant_docs = qrels.get(query_id, [])

        y_true = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]

        if sum(y_true) == 0:
            print(f"No relevant documents found for query {query_id}. Skipping recall and precision calculation.")
            print(f'Retrieved docs: {len(retrieved_docs)}')
            print(f'Relevant docs: {len(relevant_docs)}')

        y_score = [hit['_score'] for hit in search_results['hits']['hits']]
        
        precision = precision_score(y_true, [1 if score >= 0.5 else 0 for score in y_score])
        recall = recall_score(y_true, [1 if score >= 0.5 else 0 for score in y_score])
        
        ap = average_precision_score(y_true, y_score)
        
        precision_list.append(precision)
        recall_list.append(recall)
        ap_list.append(ap)
    
    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)
    mean_ap = sum(ap_list) / len(ap_list)
    
    print(f"AVG Precision: {avg_precision}")
    print(f"AVG Recall: {avg_recall}")
    print(f"Mean Average Precision (MAP): {mean_ap}")

def process_term_vectors(term_vectors):
    terms = []
    for id, doc in term_vectors.items():
        term_freqs = {term: info['term_freq'] for term, info in doc.items()}
        terms.append(term_freqs)

    terms_matrix_df = pd.DataFrame(terms).fillna(0).reindex(sorted(pd.DataFrame(terms).columns), axis=1)

    return terms_matrix_df
