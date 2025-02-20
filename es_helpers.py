from collections import defaultdict
from elasticsearch import Elasticsearch
from elasticsearch.client import IndicesClient
from elasticsearch.helpers import bulk
from es_indices import configurations
import stopwordsiso as stopwords

import spacy
import string
import time

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.max_length = 1500000

def connect_elasticsearch(host='localhost', port=9200):
    connected = False

    while not connected:
        es = Elasticsearch(hosts=[{'host': host, 'port': port, 'scheme': 'http'}], timeout=300)
        if es.ping():
            connected = True
            print('DEBUG: Connected to Elasticsearch')
        else:
            print('ERROR: Could not connect to Elasticsearch... retrying in 3 seconds')
            time.sleep(3)

    return es

def delete_index(es, index):
    try:
        es.indices.delete(index=index)
    except Exception as e:
        print(f'ERROR: An error occurred while deleting index {index}: {e}')

def create_index(es, index, delete = False):
    if delete:
        print(f'Deleting index: {index}')
        delete_index(es, index)

    settings = get_index_configuration(index)

    try:
        if not es.indices.exists(index=index):
            es.indices.create(index=index, body=settings)
            print(f'DEBUG: Created Index: {index}')
        else:
            print(f'DEBUG: Index {index} already exists.')
    except Exception as ex:
        print(f'ERROR: An error occurred while creating index: {str(ex)}')

def index_documents(es, index, documents):
    for doc in documents:
        try:
            es.index(index=index, id=doc['DOCNO'], document={
                'DOCNO': doc['DOCNO'],
                'TEXT': doc['TEXT'],
                'TITLE': doc.get('TITLE', None),
            })
            print(f"DEBUG: Document {doc['DOCNO']} indexed")
        except Exception as ex:
            print(f"ERROR: An error occurred while indexing document {doc['DOCNO']}: {str(ex)}")

def bulk_index_documents(es, index, documents):
    existing_ids = set()
    for doc in documents:
        if es.exists(index=index, id=doc['DOCNO']):
            existing_ids.add(doc['DOCNO'])

        actions = [
            {
                "_index": index,
                "_id": doc['DOCNO'],
                "_source": {
                    'DOCNO': doc['DOCNO'],
                    'TEXT': doc['TEXT'],
                    'TITLE': doc.get('TITLE', None),
                }
            }
            for doc in documents
            if doc['DOCNO'] not in existing_ids
        ]

    try:
        success, failed = bulk(es, actions)
        print(f"DEBUG: Successfully indexed {success} documents, {failed} failed")
    except Exception as ex:
        print(f"ERROR: An error occurred while during bulk indexing: {str(ex)}")
        print('Waiting 3 minutes and re-establishing connection.')
        time.sleep(60 * 3)
        es = connect_elasticsearch()
        print(f"DEBUG: Indexing docs one by one...")
        index_documents(es, index, documents)
        print('DEBUG: Done!')

def search(es, index, query, evaluate=False):
    res = es.search(index=index, body={
        "query": {
            "multi_match": {
                "query": query['TITLE'],
                "fields": ["TITLE", "TEXT^2"],
                "operator": "or",
            },
        },
        "size": 75,
    })

    if evaluate:
        with open('trec_eval/submition.txt', 'w') as f:
            for rank, hit in enumerate(res['hits']['hits'], start=1):
                doc_id = hit['_id']
                score = hit['_score']
                f.write(f"{query['NUM']} Q0 {doc_id} {rank} {score} STANDARD\n")

    return res

def get_significant_words(es, index, query):
    res = es.search(index=index, body={
        "query": {
            "multi_match": {
                "query": query['TITLE'],
                "fields": ["TITLE", "TEXT^2"],
                "operator": "or",
            },
        },
        "size": 75,
        "aggs": {
            "significant_words": {
                "significant_text": {
                    "field": "TEXT",
                    "size": 50,
                    "filter_duplicate_text": True,
                    "exclude": [*string.printable, *get_stopwords()],
                    "min_doc_count": 5,
                    "chi_square": {},
                }
            }
        }
    })

    analyzed_query_result = IndicesClient.analyze(es, index=index, text=query['TITLE'])
    analyzed_query_terms = []
    for bucket in analyzed_query_result['tokens']:
        analyzed_query_terms.append(bucket['token'])

    vocabulary = [bucket["key"] for bucket in res['aggregations']['significant_words']['buckets']]
    for term in analyzed_query_terms:
        if not term in vocabulary:
            vocabulary.append(term)

    return vocabulary

def get_terms_window(es, index, query, documents_text):
    N = 11
    top_k = 50

    analyzed_query_result = IndicesClient.analyze(es, index=index, text=query['TITLE'])
    query_terms = [bucket['token'] for bucket in analyzed_query_result['tokens']]

    term_scores = defaultdict(float)

    for doc in documents_text:
        analyzed_text = IndicesClient.analyze(es, index=index, text=doc)
        words = [bucket['token'] for bucket in analyzed_text['tokens'] 
                 if len(bucket['token']) > 2 and not any(char.isdigit() for char in bucket['token'])]

        for term in query_terms:
            indices = [i for i, w in enumerate(words) if w == term]
            if not indices:
                continue

            for idx in indices:
                start = max(0, idx - N)
                end = min(len(words), idx + N + 1)
                for i in range(start, end):
                    distance = abs(i - idx) + 1
                    term_scores[words[i]] += 1 / distance

    sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [term for term, score in sorted_terms]

def get_term_vectors(es, index, doc_id, field='TEXT'):
    try:
        term_vector = es.termvectors(
            index=index,
            id=doc_id,
            fields=[field],
            term_statistics=True,
            field_statistics=True,
            offsets=False,
            positions=False,
            payloads=False
        )

        terms = term_vector['term_vectors'][field]['terms']

        return terms
    except Exception as e:
        print(f"ERROR: An error occurred while extracting term vector for doc {doc_id}: {str(e)}")

        return None

def extract_term_vectors_from_documents(es, index, document_ids):
    all_term_vectors = {}

    for doc_id in document_ids:
        term_vector = get_term_vectors(es, index, doc_id)
        if term_vector:
            all_term_vectors[doc_id] = term_vector

    return all_term_vectors

def check_relevant_docs_indexed(es, index, qrels, query_id):
    relevant_docs = qrels.get(query_id, [])
    if not relevant_docs:
        print(f"No relevant document retrieved for Query {query_id}")
        return

    total_relevant = len(relevant_docs)
    indexed_count = 0

    for doc_id in relevant_docs:
        if es.exists(index=index, id=doc_id):
            print(f"Document {doc_id} is indexed.")
            indexed_count += 1
        else:
            print(f"Document {doc_id} is NOT indexed.")

    print(f"Relevant documents for Query {query_id}: {total_relevant}")
    print(f"Indexed relevant documents {indexed_count}")
    print(f"Not indexed relevant documents {total_relevant - indexed_count}")

def get_stopwords():
    iso_stopwords = stopwords.stopwords("en")
    custom_stoplist = load_stoplist('storage/tipster/inquery.stoplist')

    return iso_stopwords.union(custom_stoplist)

def load_stoplist(file_path):
    with open(file_path, 'r') as f:
        stopwords = [line.strip() for line in f.readlines()]
    return stopwords

def search_documents_by_ids(es, index, ids):

    res = es.search(index=index, body={
        "query": {
            "ids": {
                "values": ids
            }
        },
        "size" : len(ids)
    })

    return res

def process_text(text, max_tokens=150000):
    if text is None:
        return text

    stoplist = load_stoplist('storage/tipster/inquery.stoplist')
    doc = nlp(text)

    processed_texts = []

    num_tokens = len(doc)
    
    for i in range(0, num_tokens, max_tokens):
        shard = doc[i:min(i + max_tokens, num_tokens)]
        
        lemmas = [
            token.lemma_ for token in shard
            if token.lemma_.lower() not in stoplist and token.is_alpha
        ]
        processed_texts.extend(lemmas)

    return " ".join(processed_texts)

def get_index_configuration(index_name):
    config = configurations.get(index_name)

    if config is not None:
        return config
    else:
        raise ValueError(f"Configuration for index '{index_name}' not found.")
