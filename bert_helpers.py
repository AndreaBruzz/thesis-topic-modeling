from bertopic import BERTopic
from gensim.corpora import Dictionary
from sklearn.cluster import KMeans
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from sentence_transformers import SentenceTransformer
from umap import UMAP

import pandas as pd

def configure_bertopic(embedding_model_name="all-MiniLM-L6-v2", topwords=5, topics=51):
    embedding_model = SentenceTransformer(embedding_model_name)
    umap_model = UMAP()
    cluster_model = KMeans(n_clusters=topics)

    topic_model = BERTopic(
        top_n_words=topwords,
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=cluster_model,
    )

    return topic_model

def run_bertopic_model(topwords=5, topics=5, documents=None):
    if not documents:
        file_path = 'storage/octis/running_dataset/corpus.tsv'
        column_names = ['document', 'label']
        df = pd.read_csv(file_path, sep='\t', header=None, names=column_names)
        documents = df['document'].tolist()

    topic_model = configure_bertopic(topwords=topwords, topics=topics)
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

def get_topic_vectors(topic_model):
    return topic_model.c_tf_idf_.toarray().tolist()

def get_top_topics(topic_model, top_n):
    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info["Topic"] != -1].copy()

    return topic_info["Topic"].head(int(top_n)).tolist()

def embed_topics(topics):
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    topic_embeds = {}

    for tid, ww in topics.items():
        if not ww:
            continue
        words = [w for w, _ in ww]
        E = encoder.encode(words, convert_to_numpy=True)
        v = E.mean(axis=0)

        topic_embeds[tid] = v.astype(np.float32)

    return topic_embeds

def topic_to_text(topic_tuples):
    words = [w for w, _ in topic_tuples]

    return " ".join(words)
