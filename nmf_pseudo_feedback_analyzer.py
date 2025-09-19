from parsers import QueryParser
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer

import es_helpers
import octis_helpers
import utils
import os
import operators

def main():
    es, args = utils.setup()

    index = "tipster_45_kstem"

    query_parser = QueryParser('storage/queries/robust04.topics')
    queries = query_parser.parse_queries()

    run_id = "prf_chi_square_nmf"

    try:
        os.remove(f'storage/runs/{run_id}')
    except:
        pass

    os.makedirs(f'storage/runs/{run_id}', exist_ok=True)

    for _, query in queries.items():
        print(f"\n==========================================================")
        print(f"Processing query: {query['num']} - {query['title']}")
        print(f"==========================================================\n")

        print('Running baseline search...')
        hits = es_helpers.search(es, index, query, evaluate=True, run_id=f'{run_id}/baseline')['hits']['hits']
        doc_ids   = [h['_id'] for h in hits]
        doc_texts = [h['_source']['TEXT'] for h in hits]

        print('\nRetrieving PRF documents')
        oracle_hits = es_helpers.search(es, index, query, 75)['hits']['hits']
        oracle_documents_id   = [h['_id'] for h in oracle_hits]
        oracle_documents_text = [h['_source']['TEXT'] for h in oracle_hits]

        topics = 6
        topwords = 6

        for selected_vocabulary in ['tw', 'st']:
            print(f'\nUsing vocabulary source: {selected_vocabulary}\n')

            if selected_vocabulary == 'tw':
                vocabulary = es_helpers.get_terms_window(es, index, query, oracle_documents_text)
            elif selected_vocabulary == 'st':
                vocabulary = es_helpers.get_significant_words(es, index, query, oracle_documents_id)

            dataset = octis_helpers.create_dataset(oracle_documents_text, vocabulary)

            print('\n###### NMF MODEL ######\n')
            nmf_output, nmf_id2word = octis_helpers.run_nmf_model(dataset, topics, topwords)
            octis_helpers.display_topics(nmf_output, nmf_id2word, topwords)

            topic_vectors = octis_helpers.get_topic_vectors(nmf_output)
            id2word = nmf_id2word

            top_n = 4
            top_topic_ids = octis_helpers.get_top_topics(nmf_output, top_n)
            topic_vectors = [topic_vectors[i] for i in top_topic_ids]

            print(f"\nSelected top {top_n} topics:")
            for idx in top_topic_ids:
                top_words = [id2word[i] for i in topic_vectors[top_topic_ids.index(idx)].argsort()[-10:][::-1]]
                print(f"  Topic {idx}: {' | '.join(top_words)}")

            vectorizer = CountVectorizer(vocabulary=list(id2word.values()))
            documents_matrix = vectorizer.fit_transform(doc_texts)
            documents_vectors = documents_matrix.toarray()
            del vectorizer, documents_matrix

            topic_vectors_named = [(f"t{i}", vec) for i, vec in enumerate(topic_vectors)]
            all_rerank_configs = []

            print("\nCreating single-topic reranking configurations...")
            for name, vec in topic_vectors_named:
                print(f"  - single_{name}")
                all_rerank_configs.append((f"single_{name}", [vec]))

            print("\nCreating join-topic reranking configurations...")
            for (name1, vec1), (name2, vec2) in combinations(topic_vectors_named, 2):
                join_vec = operators.join(vec1, vec2)
                run_name = f"join_{name1}_{name2}"
                print(f"  - {run_name}")
                all_rerank_configs.append((run_name, join_vec))

            print("\nCreating meet-topic reranking configurations...")
            for comb in combinations(topic_vectors_named, 4):
                for (p1, p2) in utils.pairings_of_four(comb):
                    (name_a, vec_a), (name_b, vec_b) = p1
                    (name_c, vec_c), (name_d, vec_d) = p2
                    meet_vec = operators.meet(vec_a, vec_b, vec_c, vec_d)
                    run_name = f"meet_({name_a}_{name_b})_({name_c}_{name_d})"
                    print(f"  - {run_name}")
                    all_rerank_configs.append((run_name, [meet_vec]))

            print("\nPerforming reranking for all configurations...")
            for run_name, topics_vectors in all_rerank_configs:
                reranked = utils.rerank_documents_v2(
                    documents_vectors,
                    topics_vectors,
                    doc_ids
                )
                print(f"\nReranking completed for: {run_name}")
                utils.write_trec_run(query['num'], reranked, f'{run_name}_{selected_vocabulary}', f'storage/runs/{run_id}')

if __name__ == "__main__":
    main()
