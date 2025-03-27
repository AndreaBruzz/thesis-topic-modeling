with open('storage/tipster/inquery.stoplist', 'r') as f:
        stopwords = [line.strip() for line in f.readlines()]

configurations = {
    # Baseline with no stemming
    "tipster_45": {
        "settings": {
            "analysis": {
                "analyzer": {
                    "text_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "custom_stop"]
                    }
                },
                "filter": {
                    "custom_stop": {
                        "type": "stop",
                        "stopwords": [*stopwords, "_english_"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "docno": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "text_analyzer",
                },
                "text": {
                    "type": "text",
                    "analyzer": "text_analyzer",
                },
                'embedding_full': {
                    'type': 'dense_vector',
                    "dims": 384, # specific con all-MiniLM-L6-v2
                },
                'embedding_trunc': {
                    'type': 'dense_vector',
                    "dims": 384, # specific con all-MiniLM-L6-v2
                }
            }
        }
    },
    # Porter stemmer, aggressive normalization
    "tipster_45_porter": {
        "settings": {
            "analysis": {
                "analyzer": {
                    "text_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "custom_stop", "porter_stem"]
                    }
                },
                "filter": {
                    "custom_stop": {
                        "type": "stop",
                        "stopwords": [*stopwords, "_english_"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "docno": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "text_analyzer",
                },
                "text": {
                    "type": "text",
                    "analyzer": "text_analyzer",
                },
                'embedding_full': {
                    'type': 'dense_vector',
                    "dims": 384, # specific con all-MiniLM-L6-v2
                },
                'embedding_trunc': {
                    'type': 'dense_vector',
                    "dims": 384, # specific con all-MiniLM-L6-v2
                }
            }
        }
    },
    # Kstemmer, lighter normalization
    "tipster_45_kstem": {
        "settings": {
            "analysis": {
                "analyzer": {
                    "text_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "custom_stop", "kstem"]
                    }
                },
                "filter": {
                    "custom_stop": {
                        "type": "stop",
                        "stopwords": [*stopwords, "_english_"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "docno": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "text_analyzer",
                },
                "text": {
                    "type": "text",
                    "analyzer": "text_analyzer",
                },
                'embedding_full': {
                    'type': 'dense_vector',
                    "dims": 384, # specific con all-MiniLM-L6-v2
                },
                'embedding_trunc': {
                    'type': 'dense_vector',
                    "dims": 384, # specific con all-MiniLM-L6-v2
                }
            }
        }
    },
}
