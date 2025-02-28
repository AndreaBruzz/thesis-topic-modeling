from parsers import QueryParser, QrelsParser
from random import sample
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import argparse
import copy
import es_helpers
import os
import numpy as np
import pandas as pd
import random
import subprocess

def setup():
    random.seed(a=754)

    es = es_helpers.connect_elasticsearch()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--index", action="store_true", help="Index files")
    parser.add_argument("-d", "--delete-index", action="store_true", help="Delete index before indexing")
    parser.add_argument("-e", "--evaluate", action="store_true", help="Evaluate search engine performance")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print step by step results")
    parser.add_argument("-s", "--simulate", action="store_true", help="Simulate a random query and get only relevant docs for it")
    parser.add_argument("-t", "--tune", action="store_true", help="Find best parameters for each qrel")

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

def simulate_search(es, index, query, subset_size):
    qrels_parser = QrelsParser('storage/queries/robust04.qrels')
    qrels = qrels_parser.parse_qrels()

    document_ids = qrels[query['NUM']]

    document_ids = sample(document_ids, int(len(document_ids) * subset_size))

    return es_helpers.search_documents_by_ids(es, index, document_ids)

def run_trec_eval(index):
    try:
        os.remove(f'storage/eval/eval_{index}.txt')
    except:
        pass

    os.chdir('trec_eval')

    command = ['./trec_eval', '../storage/queries/robust04.qrels', 'submition.txt']

    with open(f'../storage/eval/eval_{index}.txt', 'w') as eval_file:
        result = subprocess.run(command, stdout=eval_file, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print('Errore durante l\'esecuzione di trec_eval:')
        print(result.stderr)
    else:
        print(f'Valutazione completata con successo. Output salvato in eval/eval_{index}.txt')

    os.chdir('..')


def select_index():
    indexes = {
        1: 'tipster_45',
        2: 'tipster_45_porter',
        3: 'tipster_45_kstem',
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

def topic_from_vector(id2word, vector, topk):
    indices = np.argsort(vector)[-topk:][::-1]
    topic = [id2word[idx] for idx in indices]

    return topic

def rerank_documents(reranking_type, documents, query, topics):
    qrels_parser = QrelsParser('storage/queries/robust04.qrels')
    qrel_docs = qrels_parser.parse_qrels(query['NUM'])

    documents_id = list(documents.keys())
    documents_text = list(documents.values())

    query_embedding = embed_text(query['TITLE'])
    topic_embeddings = embed_text(topics)

    # Quanto un topic è relativo alla query
    query_topic_similarity = cosine_similarity([query_embedding], topic_embeddings)[0]

    # Quanto un topic è relativo al documento
    topic_doc_matrix = calculate_topic_document_matrix(topics, documents_text)

    # Punteggio della relazione topic-query in base alla relazione topic-doc
    semantic_scores = []
    for i in range(topic_doc_matrix.shape[0]):
        doc_topic_probs = topic_doc_matrix[i]
        semantic_score = np.dot(doc_topic_probs, query_topic_similarity)
        semantic_scores.append(semantic_score)

    final_scores = [semantic_scores[i] for i in range(len(documents_id))]

    if (reranking_type == 3):
        n = 5
        frozen_docs_id = documents_id[:n]
        frozen_scores = final_scores[:n]
        documents_id = documents_id[n:]
        frozen_scores = final_scores[n:]

        frozen_documents = [(frozen_docs_id[i], frozen_scores[i]) for i in range(len(frozen_docs_id))]

    reranked_documents = sorted(
        [(documents_id[i], final_scores[i]) for i in range(len(documents_id))],
        key=lambda x: x[1],
        reverse=True
    )

    if (reranking_type == 2):
        return residual_ranking(reranked_documents, qrel_docs)
    elif (reranking_type == 3):
        return frozen_documents + reranked_documents
    else:
        return reranked_documents

def residual_ranking(docs, qrel_docs):
    return [doc for doc in docs if doc[0] not in set(qrel_docs['relevant']) | set(qrel_docs['non_relevant'])]

def embed_text(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    return model.encode(text)


def calculate_topic_document_matrix(topics, documents):
    topic_embeddings = embed_text(topics)
    document_embeddings = embed_text(documents)

    topic_document_matrix = cosine_similarity(document_embeddings, topic_embeddings)

    return topic_document_matrix

def print_rank(reranked_docs, ranked_docs=None, truncate=100):
    if ranked_docs is None:
        for rank, (doc_id, score) in enumerate(reranked_docs, 1):
            print(f"{rank:<5} {doc_id:<20} {round(score, 6)}")
    else:
        prev_rank_map = {doc_id: i + 1 for i, (doc_id, _) in enumerate(ranked_docs)}
        count = 0

        for rank, (doc_id, score) in enumerate(reranked_docs, 1):
            if count == truncate: break
            count += 1

            prev_rank = prev_rank_map[doc_id]

            if prev_rank == rank:
                change = "=="
            elif prev_rank < rank:
                change = f"-{rank - prev_rank}"
            else:
                change = f"+{prev_rank - rank}"

            print(f"{rank:<5} {doc_id:<20} {round(score, 6):<12} {change}")

def select_model():
    models = {
        1: 'NMF',
        2: 'BERT'
    }

    while True:
        for key, val in models.items():
            print(f'{key} - {val}')
        try:
            index = int(input('Select model: '))
            if index in models:
                return models[index]
            else:
                print('Invalid selection. Please enter a number from the list.')
        except ValueError:
            print('Invalid input. Please enter a valid number.')

def select_vocabulary():
    print()

    vocabulary_options = {
        1: 'Terms window',
        2: 'Significant terms',
    }

    while True:
        for key, val in vocabulary_options.items():
            print(f'{key} - {val}')
        try:
            return int(input('Select vocabulary: '))
        except ValueError:
            print('Invalid input. Please enter a valid number.')

def select_reranking():
    print()

    vocabulary_options = {
        1: 'No method',
        2: 'Residual Ranking',
        3: 'Frozen Ranking',
    }

    while True:
        for key, val in vocabulary_options.items():
            print(f'{key} - {val}')
        try:
            return int(input('Select reranking method: '))
        except ValueError:
            print('Invalid input. Please enter a valid number.')

def ask_oracle(res, query):
    qrels_parser = QrelsParser('storage/queries/robust04.qrels')
    qrel_docs = qrels_parser.parse_qrels(query['NUM'])

    # Ensure relevant docs are in a set for fast lookup
    relevant_doc_ids = set(qrel_docs['relevant'])

    # Make a deep copy to avoid modifying the original res
    filtered_res = copy.deepcopy(res)

    # Filter the hits in the copied response
    filtered_res['hits']['hits'] = [doc for doc in filtered_res['hits']['hits'] if doc['_id'] in relevant_doc_ids]

    return filtered_res  # Return the modified copy
