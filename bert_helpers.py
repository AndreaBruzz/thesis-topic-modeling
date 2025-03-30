from bertopic import BERTopic
from gensim.corpora import Dictionary
from hdbscan import HDBSCAN
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from sentence_transformers import SentenceTransformer
from umap import UMAP

import pandas as pd

def configure_bertopic(embedding_model_name="all-MiniLM-L6-v2", topwords=5, min_cluster_size=5, n_components=15, min_dist=0.1):
    embedding_model = SentenceTransformer(embedding_model_name)
    umap_model = UMAP(n_components=n_components, min_dist=min_dist, random_state=12)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1)

    topic_model = BERTopic(
        top_n_words=topwords,
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
    )

    return topic_model

def run_bertopic_model(topwords=5, documents=None):
    if not documents:
        file_path = 'storage/octis/running_dataset/corpus.tsv'
        column_names = ['document', 'label']
        df = pd.read_csv(file_path, sep='\t', header=None, names=column_names)
        documents = df['document'].tolist()

    topic_model = configure_bertopic(topwords=topwords)
    topic_model.fit_transform(documents)

    id2word = topic_model.vectorizer_model.get_feature_names()

    return topic_model, id2word

def evaluate_model(topic_model, dataset, topwords=5):
    all_topics = topic_model.get_topics()

    valid_topics = {}
    for topic_id, topic in all_topics.items():
        if topic_id != -1 and topic is not None:
            valid_topics[topic_id] = topic

    topics = []
    for topic_id, topic in valid_topics.items():
        topics.append([word for word, _ in topic[:topwords]])

    coherence_metric = Coherence(measure="c_v", texts=dataset.get_corpus(), topk=topwords)
    coherence_metric.corpus = dataset.get_corpus()
    coherence_metric.vocabulary = dataset.get_vocabulary()

    topics = {"topics": topics}
    coherence_score = coherence_metric.score(topics)
    print(f"Coherence Score: {coherence_score}")

    diversity_metric = TopicDiversity(topk=topwords)
    diversity_score = diversity_metric.score(topics)
    print(f"Diversity Score: {diversity_score}")

def display_topics(topic_model, include_outliers=False):
    topic_info = topic_model.get_topic_info()

    for topic_id in topic_info["Topic"]:
        if not include_outliers and topic_id == -1:
            continue
        topic_keywords = topic_model.get_topic(topic_id)
        formatted_keywords = ", ".join([f"{word} ({weight:.3f})" for word, weight in topic_keywords])
        count = topic_info.loc[topic_info["Topic"] == topic_id, "Count"].values[0]
        print(f"Topic {topic_id} ({count} docs): {formatted_keywords}")

def plot_topic_hierarchy(topic_model):
    print("Saved topics hierarchy tree at plots/bert/hierarchy.html")
    fig = topic_model.visualize_hierarchy()
    fig.write_html("storage/plots/bert/hierarchy.html")

def plot_topic_barchart(topic_model):
    print("Saved topics bar chart at plots/bert/barchart.html")
    fig = topic_model.visualize_barchart()
    fig.write_html("storage/plots/bert/barchart.html")

def get_topic_vectors(bert_output):
    topic_vectors = []
    dense_c_tf_idf = bert_output.c_tf_idf_.toarray()

    for row in dense_c_tf_idf[1:]:
        topic_vectors.append(row)

    return topic_vectors
