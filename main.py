from parsers import TipsterParser, QueryParser

import es_helpers
import octis_helpers
import random
import sys
import utils

def select_index():
    indexes = {
        1:'tipster_45_lgt', # LIGHT ENGLISH DISKS 4 AND 5
        2:'tipster_45_min', # MINIMAL ENGLISH DISKS 4 AND 5
        3:'tipster_45_reb', # REBUILT ENGLISH DISKS 4 AND 5
        4:'tipster_lgt', # LIGHT ENGLISH
        5:'tipster_min', # MINIMAL ENGLISH
    }

    for key, val in indexes.items():
        print(f'{key} - {val}')

    try:
        index = int(input('Select index: '))
    except Exception:
        index = 3

    return  indexes.get(index)

def main():
    es, args = utils.setup()

    index = select_index()

    if args.index:
        if args.delete_index:
            print(f'Deleting index: {index}')
            es_helpers.delete_index(es, index)

        es_helpers.create_index(es, index)

        leaves = utils.get_leaf_directories('storage/tipster/corpora_unzipped')

        for leaf in leaves:
            print(f'DEBUG: Processing leaf directory: {leaf}')
            parser = TipsterParser(leaf)
            parsed_documents = parser.process_all_documents()

            es_helpers.bulk_index_documents(es, index, parsed_documents)

            parser.save_stats()
    elif args.delete_index:
        print('Can\'t delete index without calling the indexing again')
        sys.exit()

    if args.evaluate:
        utils.evaluate(es, index)

    query_parser = QueryParser('storage/queries/robust04.topics')
    queries = query_parser.parse_queries()

    query = random.choice(list(queries.values()))

    if args.verbose:
        print('----------------------------------------')
        print('RANDOM QUERY: ')
        for key,val in query.items():
            print(f'{key}: {val}')
        input('Press enter to continue')

    res = es_helpers.search(es, index, query)
    documents = [hit["_source"]["TEXT"] for hit in res["hits"]["hits"]]
    document_ids = [hit['_id'] for hit in res['hits']['hits']]

    if args.verbose:
        print('----------------------------------------')
        print(f'Found {len(document_ids)} documents:')
        print(document_ids)
        input('Press enter to continue')

    term_vectors = es_helpers.extract_term_vectors_from_documents(es, index, document_ids)

    if args.verbose:
        print('----------------------------------------')
        for key,val in term_vectors.items():
            print(f'{key}:')
            for k,v in val.items():
                print(f'{k}: {v}')
        input('Press enter to continue')

    tf_matrix = utils.process_term_vectors(term_vectors)
    if args.verbose:
        print('----------------------------------------')
        print(tf_matrix)
        input('Press enter to continue')

    octis_helpers.create_dataset(documents)
    topics = octis_helpers.train_nmf_model()
    octis_helpers.display_topics(topics)

if __name__ == "__main__":
    main()
