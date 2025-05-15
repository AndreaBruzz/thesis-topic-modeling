import pymongo

def client():
    return pymongo.MongoClient("mongodb://root:password@localhost:27017/")

def get_collection(col_name, db_name, client):
    db = client[db_name]

    return db[col_name]

def add_document(collection, document):
    try:
        collection.insert_one(document)
    except pymongo.errors.PyMongoError as e:
        return

def format_data(doc_id, query_id, is_relevant, rank, notes):
    return {
        "doc_id": doc_id,
        "query_id": query_id,
        "is_relevant": is_relevant,
        "rank": rank,
        "notes": notes
    }
