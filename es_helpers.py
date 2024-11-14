from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from nltk.corpus import stopwords

import time
import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.max_length = 1500000

def connect_elasticsearch(host='localhost', port=9200):
    connected = False

    while not connected:
        es = Elasticsearch(hosts=[{'host': host, 'port': port, 'scheme': 'http'}], timeout=30)
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

def create_index(es, index):
    stoplist = load_stoplist('storage/tipster/inquery.stoplist')

    settings = get_index_configuration(index, stoplist)

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
            if index == 'test':
                es.index(index=index, id=doc['DOCNO'], document={
                    'DOCNO': doc['DOCNO'],
                    'TEXT': doc['TEXT'],
                    'TEXT_PROCESSED' : process_text(doc['TEXT']),
                    'TITLE': doc.get('TITLE', None),
                    'TITLE_PROCESSED' : process_text(doc.get('TITLE', None)),
                })
            else:
                es.index(index=index, id=doc['DOCNO'], document={
                    'DOCNO': doc['DOCNO'],
                    'TEXT': doc['TEXT'],
                    'TITLE': doc.get('TITLE', None),
                })
            print(f"DEBUG: Document {doc['DOCNO']} indexed")
        except Exception as ex:
            print(f"ERROR: An error occurred while indexing document {doc['DOCNO']}: {str(ex)}")

def bulk_index_documents(es, index, documents):
    stoplist = load_stoplist('storage/tipster/inquery.stoplist')

    existing_ids = set()
    for doc in documents:
        if es.exists(index=index, id=doc['DOCNO']):
            existing_ids.add(doc['DOCNO'])

    if index == 'test':
        actions = [
            {
                "_index": index,
                "_id": doc['DOCNO'],
                "_source": {
                    'DOCNO': doc['DOCNO'],
                    'TEXT': doc['TEXT'],
                    'TEXT_PROCESSED' : process_text(doc['TEXT']),
                    'TITLE': doc.get('TITLE', None),
                    'TITLE_PROCESSED' : process_text(doc.get('TITLE', None)),
                }
            }
            for doc in documents
            if doc['DOCNO'] not in existing_ids
        ]
    else:
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
        print('Waiting 10 minutes and re-establishing connection.')
        time.sleep(60 * 10)
        es = connect_elasticsearch()
        print(f"DEBUG: Indexing docs one by one...")
        index_documents(es, index, documents)
        print('DEBUG: Done!')

def search(es, index, query, evaluate=False):
    if index == 'test':
        res = es.search(index=index, body={
            "query": {
                "multi_match": {
                    "query": process_text(query['TITLE']),
                    "fields": ["TITLE_PROCESSED", "TEXT_PROCESSED^2"],
                    "operator": "or",
                },
            },
            "size": 100,
        })
    else:
        res = es.search(index=index, body={
            "query": {
                "multi_match": {
                    "query": query['TITLE'],
                    "fields": ["TITLE", "TEXT^2"],
                    "operator": "or",
                },
            },
            "size": 25,
        })

    if evaluate:
        with open('trec_eval/submition.txt', 'a') as f:
            for rank, hit in enumerate(res['hits']['hits'], start=1):
                doc_id = hit['_id']
                score = hit['_score']
                f.write(f"{query['NUM']} Q0 {doc_id} {rank} {score} STANDARD\n")

    return res

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
    nltk_stopwords = set(stopwords.words('english'))
    custom_stoplist = load_stoplist('storage/tipster/inquery.stoplist')

    return nltk_stopwords.union(custom_stoplist)

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
        }
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

def get_index_configuration(index_name, stoplist):
    configurations = {
        'tipster_kstem_customstop': {
            "settings": {
                "analysis": {
                    "filter": {
                        "custom_stop": {
                            "type": "stop",
                            "stopwords": stoplist
                        },
                        "kstem_filter": {
                            "type": "kstem"
                        }
                    },
                    "analyzer": {
                        "custom_kstem_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "custom_stop",
                                "kstem_filter"
                            ]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "DOCNO": {
                        "type": "keyword"
                    },
                    "TITLE": {
                        "type": "text",
                        "analyzer": "custom_kstem_analyzer"
                    },
                    "TEXT": {
                        "type": "text",
                        "analyzer": "custom_kstem_analyzer"
                    }
                }
            }
        },
        'tipster_light_english_defaultstop': {
            "settings": {
                "analysis": {
                    "filter": {
                        "english_stop": {
                            "type": "stop",
                            "stopwords": "_english_"
                        },
                        "light_english_stemmer": {
                            "type": "stemmer",
                            "language": "light_english"
                        }
                    },
                    "analyzer": {
                        "light_english_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "english_stop",
                                "light_english_stemmer"
                            ]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "DOCNO": {
                        "type": "keyword"
                    },
                    "TITLE": {
                        "type": "text",
                        "analyzer": "light_english_analyzer"
                    },
                    "TEXT": {
                        "type": "text",
                        "analyzer": "light_english_analyzer"
                    }
                }
            }
        },
        'tipster_minimal_english_customstop': {
            "settings": {
                "analysis": {
                    "filter": {
                        "custom_stop": {
                            "type": "stop",
                            "stopwords": stoplist
                        },
                        "minimal_english_stemmer": {
                            "type": "stemmer",
                            "language": "minimal_english"
                        }
                    },
                    "analyzer": {
                        "minimal_english_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "custom_stop",
                                "minimal_english_stemmer"
                            ]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "DOCNO": {
                        "type": "keyword"
                    },
                    "TITLE": {
                        "type": "text",
                        "analyzer": "minimal_english_analyzer"
                    },
                    "TEXT": {
                        "type": "text",
                        "analyzer": "minimal_english_analyzer"
                    }
                }
            }
        },
        'tipster_porter_combinedstop': {
            "settings": {
                "analysis": {
                    "filter": {
                        "combined_stop": {
                            "type": "stop",
                            "stopwords": ["_english_"] + stoplist
                        },
                        "porter_stemmer": {
                            "type": "stemmer",
                            "language": "english"
                        }
                    },
                    "analyzer": {
                        "porter_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "combined_stop",
                                "porter_stemmer"
                            ]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "DOCNO": {
                        "type": "keyword"
                    },
                    "TITLE": {
                        "type": "text",
                        "analyzer": "porter_analyzer"
                    },
                    "TEXT": {
                        "type": "text",
                        "analyzer": "porter_analyzer"
                    }
                }
            }
        },
        'tipster_kstem_ngrams': {
            "settings": {
                "analysis": {
                    "filter": {
                        "custom_stop": {
                            "type": "stop",
                            "stopwords": stoplist
                        },
                        "kstem_filter": {
                            "type": "kstem"
                        },
                        "bigram_filter": {
                            "type": "shingle",
                            "min_shingle_size": 2,
                            "max_shingle_size": 2,
                            "output_unigrams": True
                        }
                    },
                    "analyzer": {
                        "kstem_ngram_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "custom_stop",
                                "kstem_filter",
                                "bigram_filter"
                            ]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "DOCNO": {
                        "type": "keyword"
                    },
                    "TITLE": {
                        "type": "text",
                        "analyzer": "kstem_ngram_analyzer"
                    },
                    "TEXT": {
                        "type": "text",
                        "analyzer": "kstem_ngram_analyzer"
                    }
                }
            }
        },
        "test":{
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 0,
                "analysis": {
                    "filter": {
                        "custom_stop": {
                            "type": "stop",
                            "stopwords": stoplist
                        },
                        "light_english_stemmer": {
                            "type": "stemmer",
                            "name": "light_english"
                        }
                    },
                    "analyzer": {
                        "light_english": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "light_english_stemmer", "custom_stop"]
                        }
                    }
                },
            },
            "mappings": {
                    "properties": {
                        "DOCNO": {"type": "keyword"},
                        "TEXT": {"type": "text", "analyzer": "minimal_english"},
                        "TEXT_PROCESSED": {"type": "text", "analyzer": "minimal_english"},
                        "TITLE": {"type": "text", "analyzer": "minimal_english"},
                        "TITLE_PROCESSED": {"type": "text", "analyzer": "minimal_english"},
                    }
                }
        }

    }

    config = configurations.get(index_name)

    if config is not None:
        return config
    else:
        raise ValueError(f"Configuration for index '{index_name}' not found.")
