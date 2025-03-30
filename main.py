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

    if args.run_index:
        index = args.run_index
    else:
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
            subset_size = 0.7
            res = utils.simulate_search(es, index, query, subset_size)
        else:
            if args.feedback_method and args.evaluation_type:
                feedback_type = args.feedback_method
                evaluation_type = args.evaluation_type
            else:
                feedback_type, evaluation_type = utils.select_feedback()

            res = es_helpers.search(es, index, query)

            if feedback_type == 'Pseudo Feedback':
                oracle_res = es_helpers.search(es, index, query, 75)
            else:
                # Must be written better and maybe split into more functions
                oracle_res = utils.ask_oracle(res, query, feedback_type)
                if "Oracle" in feedback_type:
                    oracle_res = es_helpers.search_by_id(es, index, oracle_res)

            ranked_docs = []
            for hit in res['hits']['hits']:
                doc_id = hit['_id']
                score = hit['_score']
                ranked_docs.append((doc_id, score))

            print('Document ranking:')
            utils.print_rank(ranked_docs)

        oracle_documents = {}
        for hit in oracle_res['hits']['hits']:
            oracle_documents[hit['_id']] = hit["_source"]["text"]

        oracle_documents_text = list(oracle_documents.values())
        oracle_documents_id = list(oracle_documents.keys())

        vocabulary = []
        if args.vocab_source:
            selected_vocabulary = args.vocab_source
        else:
            selected_vocabulary = utils.select_vocabulary()

        if (selected_vocabulary == 'Terms window'):
                vocabulary = es_helpers.get_terms_window(es, index, query, oracle_documents_text)
        elif (selected_vocabulary == 'Significant terms'):
                vocabulary = es_helpers.get_significant_words(es, index, query, oracle_documents_id)

        dataset = octis_helpers.create_dataset(oracle_documents_text, vocabulary)

        topics = 6
        topwords = 6

        print('\n###### NMF MODEL ######\n')
        nmf_output, nmf_id2word = octis_helpers.run_nmf_model(dataset, topics, topwords)
        octis_helpers.evaluate_model(nmf_output, dataset, topwords)
        octis_helpers.display_topics(nmf_output, nmf_id2word, topwords)

        print('\n###### BERT MODEL ######\n')
        bert_output, bert_id2word = bert_helpers.run_bertopic_model(topwords)
        bert_helpers.evaluate_model(bert_output, dataset, topwords)
        bert_helpers.display_topics(bert_output, True)

        if args.topic_model:
            topic_model = args.topic_model
        else:
            topic_model = utils.select_model()

        if  topic_model == 'NMF':
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

        if args.embedding_type:
            embedding_type = args.embedding_type
        else:
            embedding_type = utils.select_embedding_type()

        documents_embeddings = []
        for hit in res['hits']['hits']:
            doc_id = hit['_id']
            embedding = hit["_source"][embedding_type]
            documents_embeddings.append(embedding)

        documents = {}
        for hit in res['hits']['hits']:
            documents[hit['_id']] = hit["_source"]["text"]

        reranked_docs = utils.rerank_documents(evaluation_type, ranked_docs, oracle_documents, documents, documents_embeddings, query, join_topics)
        print('\nRERANKED DOCUMENTS:')
        utils.print_rank(reranked_docs, ranked_docs)

    else:
        for query in queries.values():
            subset_size = 1
            res = utils.simulate_search(es, index, query, subset_size)

            documents = [hit["_source"]["text"] for hit in res["hits"]["hits"]]

            dataset = octis_helpers.create_dataset(documents)

            nmf_tuner = NMFTuner(dataset)
            nmf_tuner.run(metric_name='coherence')

if __name__ == "__main__":
    main()
