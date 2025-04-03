from parsers import QueryParser, QrelsParser, OracleParser
from random import sample
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from simple_term_menu import TerminalMenu

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

    parser.add_argument("-ri", "--run-index",         type=str, help="Name of the index to run experiments against.")
    parser.add_argument("-fm", "--feedback-method",   type=str, help="Feedback method to use.")
    parser.add_argument("-vc", "--vocab-source",      type=str, help="Vocabulary source for topic modeling.")
    parser.add_argument("-et", "--evaluation-type",   type=str, help="Evaluation type for reranking.")
    parser.add_argument("-em", "--embedding-type",    type=str, help="Embedding type used for re-ranking.")
    parser.add_argument("-md", "--topic-model",       type=str, help="Topic modeling method to use.")

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

    document_ids = qrels[query['num']]

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
    title = '\nSelect which Index to use:'
    options = ['tipster_45', 'tipster_45_porter', 'tipster_45_kstem']
    terminal_menu = TerminalMenu(menu_entries=options, title=title, clear_menu_on_exit=False)
    menu_entry_index = terminal_menu.show()

    return options[menu_entry_index]

def topic_from_vector(id2word, vector, topk):
    indices = np.argsort(vector)[-topk:][::-1]
    topic = [id2word[idx] for idx in indices]

    return topic

def rerank_documents(evaluation_type, ranked_docs, oracle_docs, documents, documents_embeddings, query, topics):
    qrels_parser = QrelsParser('storage/queries/robust04.qrels')
    qrel_docs = qrels_parser.parse_qrels(query['num'])

    documents_id = list(documents.keys())

    query_embedding = embed_text(query['title'])
    topic_embeddings = embed_text(topics)

    # Quanto un topic è relativo alla query
    query_topic_similarity = cosine_similarity([query_embedding], topic_embeddings)[0]

    # Quanto un topic è relativo al documento
    topic_doc_matrix = calculate_topic_document_matrix(topic_embeddings, documents_embeddings)

    # Punteggio della relazione topic-query in base alla relazione topic-doc
    semantic_scores = []
    for i in range(topic_doc_matrix.shape[0]):
        doc_topic_probs = topic_doc_matrix[i]
        semantic_score = np.dot(doc_topic_probs, query_topic_similarity)
        semantic_scores.append(semantic_score)

    docs_scores = {doc_id: semantic_scores[i] for i, doc_id in enumerate(documents_id)}

    if evaluation_type == 'Residual Ranking':
        reranked_documents = sorted(
            [(doc_id, docs_scores[doc_id]) for doc_id in documents_id],
            key=lambda x: x[1],
            reverse=True
        )
        return residual_ranking(reranked_documents, qrel_docs)
    elif evaluation_type == 'Frozen Ranking':
        oracle_docs_ids = list(oracle_docs.keys())
        
        non_frozen_docs = [doc_id for doc_id in documents_id if doc_id not in oracle_docs_ids]
        non_frozen_reranked = sorted(
            [(doc_id, docs_scores[doc_id]) for doc_id in non_frozen_docs],
            key=lambda x: x[1],
            reverse=True
        )

        final_ranking = []
        for doc in ranked_docs:
            if doc[0] in oracle_docs_ids:
                final_ranking.append((doc[0], docs_scores[doc[0]]))
            else:
                final_ranking.append(non_frozen_reranked.pop(0))

        return final_ranking
    else:
        reranked_documents = sorted(
            [(doc_id, docs_scores[doc_id]) for doc_id in documents_id],
            key=lambda x: x[1],
            reverse=True
        )
        return reranked_documents

def residual_ranking(docs, qrel_docs):
    return [doc for doc in docs if doc[0] not in set(qrel_docs['relevant']) | set(qrel_docs['non_relevant'])]

def embed_text(text, is_documents = False):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = []
    if is_documents:
        for doc in text:
            embedding = get_document_embedding(doc, model)
            embeddings.append(embedding)
        return embeddings

    return model.encode(text)

def embed_text_trunc(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    return model.encode(text)

def get_document_embedding(document, model, max_words=95):
    words = document.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)

    chunk_embeddings = model.encode(chunks)

    # The final embedding is the avg of all the chunks
    document_embedding = np.mean(chunk_embeddings, axis=0)
    return document_embedding

def calculate_topic_document_matrix(topic_embeddings, documents_embeddings):
    return cosine_similarity(documents_embeddings, topic_embeddings)

def print_rank(reranked_docs, ranked_docs=None, truncate=100):
    count = 0

    if ranked_docs is None:
        for rank, (doc_id, score) in enumerate(reranked_docs, 1):
            if count == truncate: break
            count += 1

            print(f"{rank:<5} {doc_id:<20} {round(score, 6)}")
    else:
        prev_rank_map = {doc_id: i + 1 for i, (doc_id, _) in enumerate(ranked_docs)}

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
    title = '\nSelect which Model to use:'
    options = ['NMF', 'BERT']
    terminal_menu = TerminalMenu(menu_entries=options, title=title, clear_menu_on_exit=False)
    menu_entry_index = terminal_menu.show()

    return options[menu_entry_index]

def select_vocabulary():
    title = '\nSelect vocabulary:'
    options = ['Terms window', 'Significant terms']
    terminal_menu = TerminalMenu(menu_entries=options, title=title, clear_menu_on_exit=False)
    menu_entry_index = terminal_menu.show()

    return options[menu_entry_index]

def select_feedback():
    title = '\nSelect Feedback type:'
    options_feedback = ['Pseudo Feedback', 'Explicit Feedback from Qrels', 'Explicit Feedback from Oracle']
    terminal_menu = TerminalMenu(menu_entries=options_feedback, title=title, clear_menu_on_exit=False)
    menu_entry_index_feedback = terminal_menu.show()

    if menu_entry_index_feedback == 0:
        evaluation_type = 'No method'
    elif menu_entry_index_feedback == 1:
        title = '\nSelect Evaluation type:'
        options_evaluation = ['Residual Ranking', 'Frozen Ranking']
        terminal_menu = TerminalMenu(menu_entries=options_evaluation, title=title, clear_menu_on_exit=False)
        menu_entry_index_evaluation = terminal_menu.show()

        evaluation_type = options_evaluation[menu_entry_index_evaluation]
    else:
        evaluation_type = 'Residual Ranking'

    return options_feedback[menu_entry_index_feedback], evaluation_type

def ask_oracle(res, query, feedback_type):
    if "Qrels" in feedback_type:
        qrels_parser = QrelsParser('storage/queries/robust04.qrels')
        qrel_docs = qrels_parser.parse_qrels(query['num'])

        relevant_doc_ids = set(qrel_docs['relevant'])

        filtered_res = copy.deepcopy(res) # Make a deep copy to avoid modifying the original res
        filtered_res['hits']['hits'] = [doc for doc in filtered_res['hits']['hits'] if doc['_id'] in relevant_doc_ids]

        return filtered_res

    elif "Oracle" in feedback_type:
        oracle_parser = OracleParser('storage/queries/robust04.oracle')

        if not os.path.exists('storage/queries/robust04.oracle'):
            oracle_parser.create_oracle('storage/queries/robust04.qrels')

        oracle_docs = oracle_parser.parse_oracle(query['num'])

        return oracle_docs

def select_embedding_type():
    title = '\nSelect Embedding type:'
    options = ['embedding_full', 'embedding_trunc']
    terminal_menu = TerminalMenu(menu_entries=options, title=title, clear_menu_on_exit=False)
    menu_entry_index = terminal_menu.show()

    return options[menu_entry_index]

def select_topics_for_reranking():
    title = '\nSelect Topics to use:'
    options = ['Meet Topics', 'Join Topics']
    terminal_menu = TerminalMenu(menu_entries=options, title=title, clear_menu_on_exit=False)
    menu_entry_index = terminal_menu.show()

    return options[menu_entry_index]
