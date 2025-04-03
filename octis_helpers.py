from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.models.NMF import NMF
from octis.preprocessing.preprocessing import Preprocessing
from nltk.tokenize import word_tokenize

import nltk
import numpy as np
import string
import es_helpers
import os

def create_dataset(documents, vocabulary, dataset_folder='storage/octis/dataset'):
    os.makedirs(dataset_folder, exist_ok=True)
    corpus_path = os.path.join(dataset_folder, "corpus.tsv")

    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in documents:
            doc = ' '.join(word_tokenize(doc.replace('\n', ' ')))
            f.write(f"{doc}\ttrain\n")

    vocabulary_path = os.path.join(dataset_folder, "vocabulary.txt")
    with open(vocabulary_path, "w", encoding="utf-8") as f:
        for word in vocabulary:
            f.write(f"{word}\n")

    dataset = Dataset()
    dataset.load_custom_dataset_from_folder('storage/octis/dataset')

    custom_stopwords = es_helpers.get_stopwords()

    preprocessor = Preprocessing(
        vocabulary=vocabulary,
        max_features=None,
        remove_punctuation=True,
        punctuation=string.punctuation,
        lemmatize=True,
        stopword_list=list(custom_stopwords),
        min_chars=2,
        min_words_docs=0,
        min_df=0.05,
        max_df=0.81,
        split=False,
    )

    dataset = preprocessor.preprocess_dataset(documents_path='storage/octis/dataset/corpus.tsv')
    dataset.save('storage/octis/running_dataset')

    print('DEBUG: dataset created at storage/octis/running_dataset')

    return dataset

def run_nmf_model(dataset, topics=10, topwords=5):
    nmf_model = NMF(num_topics=topics, use_partitions=False)

    hyperparameters = {
        'chunksize': 100,
        'passes': 7,
        'w_max_iter': 460,
        'h_max_iter': 164,
    }

    nmf_output = nmf_model.train_model(dataset, hyperparameters=hyperparameters, top_words=topwords)

    # Needed to know the exact mapping in order to reconvert topics vectors
    id2word = nmf_model.id2word

    return nmf_output, id2word 

def display_topics(nmf_output, id2word, topwords):
    topics = nmf_output['topics']
    topic_word_matrix = nmf_output['topic-word-matrix']
    topic_document_matrix = nmf_output['topic-document-matrix']

    dominant_doc_counts = count_dominant_docs(topic_document_matrix)

    for id, _ in enumerate(topics):
        word_indices = np.argsort(-topic_word_matrix[id])[:topwords]
        
        words_with_weights = [f"{id2word[idx]} ({topic_word_matrix[id, idx]:.4f})" for idx in word_indices]
        topic_line = ", ".join(words_with_weights)
        
        print(f"Topic {id} ({dominant_doc_counts[id]} docs): {topic_line}")

    train_avg_weights = nmf_output['topic-document-matrix'].mean(axis=1)

    print("\nAverage topic weights:", train_avg_weights)

def evaluate_model(nmf_output, dataset, topwords=5):
    coherence_metric = Coherence(measure='c_v', texts=dataset.get_corpus(), topk=topwords)
    coherence_score = coherence_metric.score(nmf_output)
    print(f"Coherence Score: {coherence_score}")

    diversity_metric = TopicDiversity(topk=topwords)
    diversity_score = diversity_metric.score(nmf_output)
    print(f"Diversity Score: {diversity_score}")

def get_topic_vectors(nmf_output):
    topic_word_matrix = nmf_output['topic-word-matrix']
    topic_vectors = []

    for i, topic_vec in enumerate(topic_word_matrix):
        topic_vectors.append(topic_vec)

    return topic_vectors

def count_dominant_docs(topic_document_matrix):
    dominant_topic_per_doc = np.argmax(topic_document_matrix, axis=0)
    topic_counts = np.bincount(dominant_topic_per_doc, minlength=topic_document_matrix.shape[0])

    return topic_counts
