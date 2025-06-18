from parsers import QueryParser, QrelsParser
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer

import es_helpers
import octis_helpers
import utils
import os
import operators

def main():
    es, args = utils.setup()

    index = args.run_index

    query_parser = QueryParser('storage/queries/robust04.topics')
    qrels_parser = QrelsParser('storage/queries/robust04.qrels')
    
    queries = query_parser.parse_queries()
    qrels = qrels_parser.parse_qrels()
    
    try:
        os.remove(f'storage/eval/eval_{index}.txt')
    except:
        pass
    
    for _, query in queries.items():
        print(f"\n==========================================================")
        print(f"Processing query: {query['num']} - {query['title']}")
        print(f"==========================================================\n")

        print('Running baseline search...\n')
        res = es_helpers.search(es, index, query, 10000, True, 'baseline')

        ranked_docs = []
        for hit in res['hits']['hits']:
            doc_id = hit['_id']
            score = hit['_score']
            ranked_docs.append((doc_id, score))

        print('Document ranking:')
        utils.print_rank(ranked_docs)

        print('\nRetrieving oracle documents for topic modeling...')
        oracle_res = es_helpers.search(es, index, query, 75)

        oracle_documents = {}
        for hit in oracle_res['hits']['hits']:
            oracle_documents[hit['_id']] = hit["_source"]["text"]

        oracle_documents_text = list(oracle_documents.values())
        oracle_documents_id = list(oracle_documents.keys())

        documents = {}
        for hit in res['hits']['hits']:
            documents[hit['_id']] = hit["_source"]["text"]

        vocabulary = []
        selected_vocabulary = args.vocab_source

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

        topic_vectors = octis_helpers.get_topic_vectors(nmf_output)
        id2word = nmf_id2word

        top_n=4
        top_topic_ids = octis_helpers.get_top_topics(nmf_output, top_n)
        topic_vectors = [topic_vectors[i] for i in top_topic_ids]

        print(f"\nSelected top {top_n} topics:")
        for idx in top_topic_ids:
            top_words = [id2word[i] for i in topic_vectors[top_topic_ids.index(idx)].argsort()[-10:][::-1]]
            print(f"  Topic {idx}: {' | '.join(top_words)}")

        documents_vectors = []
        if args.topic_model == 'BERT':
            for hit in res['hits']['hits']:
                doc_id = hit['_id']
                embedding = hit["_source"][args.embedding_type]
                documents_vectors.append(embedding)
        elif args.topic_model == 'NMF':
            vectorizer = CountVectorizer(vocabulary=list(id2word.values()))
            documents_matrix = vectorizer.fit_transform(list(documents.values()))
            documents_vectors = documents_matrix.toarray()

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
            names, vectors = zip(*comb)
            meet_vec = operators.meet(*vectors)
            run_name = f"meet_{'_'.join(names)}"
            print(f"  - {run_name}")
            all_rerank_configs.append((run_name, [meet_vec]))

        print("\nPerforming reranking for all configurations...")
        for run_name, topics_vectors in all_rerank_configs:
            reranked = utils.rerank_documents_v2(
                documents_vectors,
                topics_vectors,
                documents=documents
            )   
            print(f"\nReranking completed for: {run_name}")
            utils.print_rank(reranked, ranked_docs)
            utils.write_trec_run(query['num'], reranked, run_name)

if __name__ == "__main__":
    main()
