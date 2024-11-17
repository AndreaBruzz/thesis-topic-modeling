from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.models.NMF import NMF
from octis.preprocessing.preprocessing import Preprocessing
from nltk.tokenize import word_tokenize

import nltk
import string
import es_helpers
import os
import pandas as pd

def create_dataset(documents, dataset_folder='storage/octis/dataset'):
    os.makedirs(dataset_folder, exist_ok=True)
    corpus_path = os.path.join(dataset_folder, "corpus.tsv")

    nltk.download('punkt')
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in documents:
            doc = ' '.join(word_tokenize(doc.replace('\n', ' ')))
            f.write(f"{doc}\ttrain\n")

    vocabulary = set(word for doc in documents for word in doc.split() if len(word) > 3 and not word.isdigit())

    vocabulary_path = os.path.join(dataset_folder, "vocabulary.txt")
    with open(vocabulary_path, "w", encoding="utf-8") as f:
        for word in sorted(vocabulary):
            f.write(f"{word}\n")

    dataset = Dataset()
    dataset.load_custom_dataset_from_folder('storage/octis/dataset')

    custom_stopwords = es_helpers.get_stopwords()

    preprocessor = Preprocessing(
        vocabulary=None,
        max_features=None,
        remove_punctuation=True,
        punctuation=string.punctuation,
        lemmatize=True,
        stopword_list=list(custom_stopwords),
        min_chars=1,
        min_words_docs=0,
        min_df=0.05,
        max_df=0.85
    )

    dataset = preprocessor.preprocess_dataset(documents_path='storage/octis/dataset/corpus.tsv')
    dataset.save('storage/octis/running_dataset')

    print('DEBUG: dataset created at storage/octis/running_dataset')

    return dataset

def run_nmf_model(dataset, topics=10, topwords=5):
    nmf_model = NMF(num_topics=topics, random_state=754)
    nmf_output = nmf_model.train_model(dataset, top_words=topwords)

    return nmf_output

def display_topics(nmf_output):
    topics = nmf_output['topics']
    for id, topic in enumerate(topics):
        print(f"Topic {id}: {topic}")

def evaluate_model(nmf_output, dataset, topwords=5):
    coherence_metric = Coherence(texts=dataset.get_corpus(), topk=topwords)
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
