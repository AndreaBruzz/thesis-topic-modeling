from elasticsearch import Elasticsearch

def connect_elasticsearch(host='localhost', port=9200):
    es = Elasticsearch(hosts=[{'host': host, 'port': port, 'scheme': 'http'}])
    if es.ping():
        print('Connected to Elasticsearch')
    else:
        print('Could not connect to Elasticsearch')
    return es

def create_index(es, index, settings=None):
    if settings is None:
        settings = {
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "DOCNO": {
                        "type": "keyword"
                    },
                    "TEXT": {
                        "type": "text"
                    }
                }
            }
        }

    try:
        if not es.indices.exists(index=index):
            es.indices.create(index=index, settings=settings["settings"], mappings=settings["mappings"])
            print(f'DEBUG: Created Index: {index}')
        else:
            print(f'Index {index} already exists.')
    except Exception as ex:
        print(f'Error creating index: {str(ex)}')

def index_documents(es, index, documents):
    for doc in documents:
        try:
            es.index(index=index, id=doc['DOCNO'], document=doc)
            print(f"DEBUG: Document {doc['DOCNO']} indexed")
        except Exception as ex:
            print(f"Error indexing document {doc['DOCNO']}: {str(ex)}")
