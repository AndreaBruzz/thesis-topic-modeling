from itertools import combinations
from parsers import TipsterParser, QueryParser
from tuners import NMFTuner
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

import bert_helpers
import es_helpers
import octis_helpers
import operators
import random
import sys
import utils

def main():
    es, args = utils.setup()

    # --- Index selection ---
    index = args.run_index or utils.select_index()

    # --- (Re)indexing phase ---
    if args.index:
        es_helpers.create_index(es, index, args.delete_index)
        for leaf in utils.get_leaf_directories('storage/tipster/corpora_unzipped'):
            print(f'DEBUG: Processing leaf directory: {leaf}')
            parser = TipsterParser(leaf)
            parsed_documents = parser.process_all_documents()
            es_helpers.bulk_index_documents(es, index, parsed_documents)
    elif args.delete_index:
        print("Can't delete index without calling the indexing again")
        sys.exit()

    # --- Evaluation on index (optional) ---
    if args.evaluate:
        utils.evaluate(es, index)
        utils.run_trec_eval(index)

    # --- Load queries ---
    query_parser = QueryParser('storage/queries/robust04.topics')
    queries = query_parser.parse_queries()

    if not args.tune:
        # --- Pick 1 random query ---
        query = random.choice(list(queries.values()))
        print("\n==========================================================")
        print(f"Processing query: {query['num']} - {query['title']}")
        print("==========================================================\n")

        # --- Feedback & evaluation configuration (available in both paths) ---
        if args.feedback_method and args.evaluation_type:
            feedback_type = args.feedback_method
            evaluation_type = args.evaluation_type
        else:
            feedback_type, evaluation_type = utils.select_feedback()

        # --- Retrieve results (simulate or real) ---
        if args.simulate:
            res = utils.simulate_search(es, index, query, subset_size=0.7)
        else:
            res = es_helpers.search(es, index, query)

        # --- Oracle / PRF set ---
        if feedback_type == 'Pseudo Feedback':
            oracle_res = es_helpers.search(es, index, query, 75)
        else:
            oracle_res_tmp = utils.ask_oracle(res, query, feedback_type)
            oracle_res = es_helpers.search_by_id(es, index, oracle_res_tmp) if "Oracle" in feedback_type else oracle_res_tmp

        # --- Baseline ranking (id, score) ---
        ranked_docs = [(h['_id'], h['_score']) for h in res['hits']['hits']]
        print('Document ranking:')
        utils.print_rank(ranked_docs)

        doc_ids   = [h['_id'] for h in res['hits']['hits']]
        doc_texts = [h['_source']['text'] for h in res['hits']['hits']]

        oracle_documents = {h['_id']: h['_source']['text'] for h in oracle_res['hits']['hits']}

        # --- Vocabulary source selection ---
        selected_vocabulary = args.vocab_source or utils.select_vocabulary()
        if selected_vocabulary == 'Terms window':
            vocabulary = es_helpers.get_terms_window(es, index, query, list(oracle_documents.values()))
        elif selected_vocabulary == 'Significant terms':
            vocabulary = es_helpers.get_significant_words(es, index, query, list(oracle_documents.keys()))
        else:
            vocabulary = []

        # --- Build OCTIS dataset (for topic models) ---
        dataset = octis_helpers.create_dataset(list(oracle_documents.values()), vocabulary)

        topics = 6
        topwords = 6

        # --- NMF ---
        print('\n###### NMF MODEL ######\n')
        nmf_output, nmf_id2word = octis_helpers.run_nmf_model(dataset, topics, topwords)
        octis_helpers.evaluate_model(nmf_output, dataset, topwords)
        octis_helpers.display_topics(nmf_output, nmf_id2word, topwords)

        # --- BERTopic ---
        print('\n###### BERT MODEL ######\n')
        bert_output, _ = bert_helpers.run_bertopic_model(topwords)
        bert_helpers.evaluate_model(bert_output, dataset, topwords)
        bert_helpers.display_topics(bert_output, True)

        # --- Choose topic model for reranking pipeline ---
        topic_model = args.topic_model or utils.select_model()

        top_n = 4
        if topic_model == 'NMF':
            topic_vectors_full = octis_helpers.get_topic_vectors(nmf_output)
            top_topic_ids = octis_helpers.get_top_topics(nmf_output, top_n)

            vectorizer = CountVectorizer(vocabulary=list(nmf_id2word.values()))
            documents_vectors = vectorizer.fit_transform(doc_texts).toarray()

            topic_vectors = [topic_vectors_full[i] for i in top_topic_ids]
        else:
            top_topic_ids = bert_helpers.get_top_topics(bert_output, top_n)

            topic_texts = [bert_helpers.topic_to_text(bert_output.get_topic(tid)) for tid in top_topic_ids]
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            topic_vectors = encoder.encode(topic_texts, normalize_embeddings=True)

            embedding_type = args.embedding_type or utils.select_embedding_type()
            documents_vectors = [h["_source"][embedding_type] for h in res['hits']['hits']]

        # --- Choose combinations type (join/meet) for reranking configs ---
        topics_type = args.topics_type or utils.select_topics_for_reranking()

        topic_vectors_named = [(f"t{i}", vec) for i, vec in enumerate(topic_vectors)]
        all_rerank_configs = []

        # Build rerank configurations
        if topics_type == 'join':
            print("\nCreating join-topic reranking configurations...")
            for (name1, vec1), (name2, vec2) in combinations(topic_vectors_named, 2):
                join_vec = operators.join(vec1, vec2)
                run_name = f"join_{name1}_{name2}"
                print(f"  - {run_name}")

                # Pretty print (words if NMF; otherwise a compact note)
                seq = join_vec if isinstance(join_vec, (list, tuple)) else [join_vec]
                for u, vec in enumerate(seq, start=1):
                    if topic_model == 'NMF':
                        top_words = [nmf_id2word[i] for i in vec.argsort()[-10:][::-1]]
                        print(f"  u{u}: {' | '.join(top_words)}")
                    else:
                        print(f"  u{u}: [embedding vector of dim {len(vec)}]")
                all_rerank_configs.append((run_name, seq))

        elif topics_type == 'meet':
            print("\nCreating meet-topic reranking configurations...")
            for comb in combinations(topic_vectors_named, 4):
                for (p1, p2) in utils.pairings_of_four(comb):
                    (name_a, vec_a), (name_b, vec_b) = p1
                    (name_c, vec_c), (name_d, vec_d) = p2
                    meet_vec = operators.meet(vec_a, vec_b, vec_c, vec_d)
                    run_name = f"meet_({name_a}_{name_b})_({name_c}_{name_d})"
                    print(f"  - {run_name}")

                    if topic_model == 'NMF':
                        top_words = [nmf_id2word[i] for i in meet_vec.argsort()[-10:][::-1]]
                        print(f"  v: {' | '.join(top_words)}")
                    else:
                        print(f"  v: [embedding vector of dim {len(meet_vec)}]")

                    all_rerank_configs.append((run_name, [meet_vec]))

        # --- Reranking for all configurations (supports Residual/Frozen) ---
        for run_name, themes in all_rerank_configs:
            reranked_docs = utils.rerank_documents_v2(
                documents_vectors,
                themes,
                doc_ids,
                evaluation_type=evaluation_type,
                ranked_docs=ranked_docs,
                oracle_docs=oracle_documents,
                query_num=query['num']
            )
            print(f'\nRERANKED DOCUMENTS for {run_name}:')
            utils.print_rank(reranked_docs, ranked_docs)

    else:
        # --- Tuning mode over all queries ---
        for query in queries.values():
            res = utils.simulate_search(es, index, query, subset_size=1)
            documents = [h["_source"]["text"] for h in res["hits"]["hits"]]
            dataset = octis_helpers.create_dataset(documents)
            nmf_tuner = NMFTuner(dataset)
            nmf_tuner.run(metric_name='coherence')

if __name__ == "__main__":
    main()
