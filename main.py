from parsers import TipsterParser, QueryParser

import es_helpers
import octis_helpers
import operators
import random
import sys
import utils

def main():
    es, args = utils.setup()
    index = utils.select_index()

    if args.index:
        es_helpers.create_index(es, index, args.delete_index)

        leaves = utils.get_leaf_directories('storage/tipster/corpora_unzipped')

        for leaf in leaves:
            print(f'DEBUG: Processing leaf directory: {leaf}')
            parser = TipsterParser(leaf)
            parsed_documents = parser.process_all_documents()

            es_helpers.bulk_index_documents(es, index, parsed_documents)
    elif args.delete_index:
        print('Can\'t delete index without calling the indexing again')
        sys.exit()

    if args.evaluate:
        utils.evaluate(es, index)
        utils.run_trec_eval(index)

    query_parser = QueryParser('storage/queries/robust04.topics')
    queries = query_parser.parse_queries()
    query = random.choice(list(queries.values()))

    if args.verbose:
        print('----------------------------------------')
        print('RANDOM QUERY: ')
        for key,val in query.items():
            print(f'{key}: {val}')
        input('Press enter to continue')

    if args.simulate:
        # Qui prendo solo documenti rilevanti (70%)
        subset_size = 0.7
        res = utils.simulate_search(es, index, query, subset_size)
    else:
        res = es_helpers.search(es, index, query)

    documents = [hit["_source"]["TEXT"] for hit in res["hits"]["hits"]]
    document_ids = [hit['_id'] for hit in res['hits']['hits']]

    if args.verbose:
        print('----------------------------------------')
        print(f'Found {len(document_ids)} documents:')
        print(document_ids)
        input('Press enter to continue')

    # ------------------------------------------------------------------------------------------
    # This section is not usefull
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
    # ------------------------------------------------------------------------------------------

    dataset = octis_helpers.create_dataset(documents)
    nmf_output = octis_helpers.run_nmf_model(dataset)
    octis_helpers.evaluate_model(nmf_output, dataset)

    octis_helpers.display_topics(nmf_output)

    topic_vectors = octis_helpers.get_topic_vectors(nmf_output)

    print('JOIN:')
    join_result = operators.join(topic_vectors[0], topic_vectors[1])
    for vector in join_result:
        print(utils.topic_from_vector(vector, 5))

    print('MEET:')
    meet_result = operators.meet(topic_vectors[0], topic_vectors[1], topic_vectors[2], topic_vectors[3])
    print(utils.topic_from_vector(meet_result, 5))

if __name__ == "__main__":
    main()
