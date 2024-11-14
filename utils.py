from parsers import QueryParser, QrelsParser
from random import sample

import argparse
import es_helpers
import os
import nltk
import pandas as pd
import random
import subprocess
import sys

def setup(index):
    random.seed(a=754)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    try:
        os.remove("trec_eval/submition.txt")
        os.remove(f'storage/eval/eval_{index}.txt')
    except:
        pass

    es = es_helpers.connect_elasticsearch()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--index", action="store_true", help="Index files")
    parser.add_argument("-d", "--delete-index", action="store_true", help="Delete index before indexing")
    parser.add_argument("-e", "--evaluate", action="store_true", help="Evaluate search engine performance")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print step by step results")
    parser.add_argument("-s", "--simulate", action="store_true", help="Simulate a random query and get only relevant docs for it")

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
    query_parser = QueryParser('storage/queries/robust04.topics')
    qrels_parser = QrelsParser('storage/queries/robust04.qrels')
    
    queries = query_parser.parse_queries()
    qrels = qrels_parser.parse_qrels()
    
    for query_id, query in queries.items():
        search_results = es_helpers.search(es, index, query, True)
        
        retrieved_docs = [hit['_id'] for hit in search_results['hits']['hits']]
        
        relevant_docs = qrels.get(query_id, [])
        if len(relevant_docs) != 0:
            y_true = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]

            if sum(y_true) == 0:
                print(f"No relevant documents found for query {query_id}. Skipping recall and precision calculation.")
                print(f'Retrieved docs: {len(retrieved_docs)}')
                print(f'Relevant docs: {len(relevant_docs)}')
                continue

def process_term_vectors(term_vectors):
    terms = []
    for id, doc in term_vectors.items():
        term_freqs = {term: info['term_freq'] for term, info in doc.items()}
        terms.append(term_freqs)

    terms_matrix_df = pd.DataFrame(terms).fillna(0).reindex(sorted(pd.DataFrame(terms).columns), axis=1)

    return terms_matrix_df

def simulate_search(es, index, query):
    qrels_parser = QrelsParser('storage/queries/robust04.qrels')
    qrels = qrels_parser.parse_qrels()

    document_ids = qrels[query['NUM']]
    document_ids = sample(document_ids, int(len(document_ids) * 0.7))

    return es_helpers.search_documents_by_ids(es, index, document_ids)

def run_trec_eval(index):
    os.chdir('trec_eval')

    command = ['./trec_eval', '../storage/queries/robust04.qrels', 'submition.txt']

    with open(f'../storage/eval/eval_{index}.txt', 'w') as eval_file:
        result = subprocess.run(command, stdout=eval_file, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print('Errore durante l\'esecuzione di trec_eval:')
        print(result.stderr)
    else:
        print('Valutazione completata con successo. Output salvato in eval.txt.')

    os.chdir('..')


def select_index():
    indexes = {
        1: 'tipster_kstem_ngrams',
        2: 'tipster_porter_combinedstop',
        3: 'tipster_minimal_english_customstop',
        4: 'tipster_light_english_defaultstop',
        5: 'tipster_kstem_customstop',
        6: 'test'
    }

    while True:
        for key, val in indexes.items():
            print(f'{key} - {val}')
        try:
            index = int(input('Select index: '))
            if index in indexes:
                return indexes[index]
            else:
                print('Invalid selection. Please enter a number from the list.')
        except ValueError:
            print('Invalid input. Please enter a valid number.')

