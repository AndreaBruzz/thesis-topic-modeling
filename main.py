from itertools import combinations
from parsers import TipsterParser, QueryParser
from tuners import NMFTuner

import bert_helpers
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

    if not args.tune:
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

            ranked_docs = []
            for hit in res['hits']['hits']:
                doc_id = hit['_id']
                score = hit['_score']
                ranked_docs.append((doc_id, score))

            print('Document ranking:')
            utils.print_rank(ranked_docs)

        documents = {}
        for hit in res['hits']['hits']:
            documents[hit['_id']] = hit["_source"]["TEXT"]
        
        documents_text = list(documents.values())
        document_ids = list(documents.keys())
        scores = [hit['_score'] for hit in res['hits']['hits']]

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

        dataset = octis_helpers.create_dataset(documents_text)

        topics = 6
        topwords = 6

        print('\n###### NMF MODEL ######\n')
        nmf_output, nmf_id2word = octis_helpers.run_nmf_model(dataset, topics, topwords)
        octis_helpers.evaluate_model(nmf_output, topwords)
        octis_helpers.display_topics(nmf_output, nmf_id2word, topwords)

        print('\n###### BERT MODEL ######\n')
        bert_output, bert_id2word = bert_helpers.run_bertopic_model(topwords)
        bert_helpers.evaluate_model(bert_output, dataset, topwords)
        bert_helpers.display_topics(bert_output)
        # bert_helpers.plot_topic_barchart(bert_output)
        # bert_helpers.plot_topic_hierarchy(bert_output)

        if utils.select_model() == 1:
            topic_vectors = octis_helpers.get_topic_vectors(nmf_output)
            id2word = nmf_id2word
        else:
            topic_vectors = bert_helpers.get_topic_vectors(bert_output)
            id2word = bert_id2word

        join_topic_vectors = []
        for v1, v2 in combinations(topic_vectors, 2):
            join_topic_vectors.extend(operators.join(v1, v2))

        join_topics = []
        print('JOIN:')
        for topic in join_topic_vectors:
            join_topics.append(utils.topic_from_vector(id2word, topic, topwords))
            print(utils.topic_from_vector(id2word, topic, topwords))

        meet_topic_vectors = []
        for v1, v2, v3, v4 in combinations(topic_vectors, 4):
            meet_topic_vectors.append(operators.meet(v1, v2, v3, v4))

        meet_topics = []
        print('MEET:')
        for topic in meet_topic_vectors:
            meet_topics.append(utils.topic_from_vector(id2word, topic, topwords))
            print(utils.topic_from_vector(id2word, topic, topwords))

        reranked_docs = utils.rerank_documents(documents, scores, query['DESC'], join_topics, alpha=1, beta=1)
        print('\nRERANKED DOCUMENTS:')
        utils.print_rank(reranked_docs, ranked_docs)

    else:
        for query in queries.values():
            subset_size = 1
            res = utils.simulate_search(es, index, query, subset_size)

            documents = [hit["_source"]["TEXT"] for hit in res["hits"]["hits"]]

            dataset = octis_helpers.create_dataset(documents)

            nmf_tuner = NMFTuner(dataset)
            nmf_tuner.run(metric_name='coherence')

if __name__ == "__main__":
    main()
