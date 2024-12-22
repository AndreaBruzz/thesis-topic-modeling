import matplotlib.pyplot as plt
import os

from collections import defaultdict
from octis.models.NMF import NMF
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity


class NMFTuner:
    def __init__(self, dataset, random_state=754):
        self.dataset = dataset
        self.random_state = random_state
        self.results = []
        self.docs_number = len(dataset._Dataset__corpus)

    def train_model(self, num_topics, topwords, chunksize=100, passes=7, w_max_iter=460, h_max_iter=164):
        nmf_model = NMF(num_topics=num_topics, random_state=self.random_state)

        hyperparameters = {
            'chunksize': chunksize,
            'passes': passes,
            'w_max_iter': w_max_iter,
            'h_max_iter': h_max_iter,
        }

        return nmf_model.train_model(self.dataset, hyperparameters=hyperparameters, top_words=topwords)

    def evaluate_model(self, nmf_output, topwords):
        try:
            coherence_metric = Coherence(measure='c_v', topk=topwords)
            coherence_score = coherence_metric.score(nmf_output)

            diversity_metric = TopicDiversity(topk=topwords)
            diversity_score = diversity_metric.score(nmf_output)
        except:
            coherence_score = 0
            diversity_score = 0

        return {'coherence': coherence_score, 'diversity': diversity_score}

    def tune_parameters(self, parameter_grid):
        for params in parameter_grid:
            print(f"Evaluating parameters: {params}")
            nmf_output = self.train_model(**params)
            metrics = self.evaluate_model(nmf_output, topwords=params.get('topwords', 5))
            self.results.append((params, metrics))

    def get_best_params(self, metric_name):
        return max(self.results, key=lambda x: x[1][metric_name])[0]

    def plot_results(self, metric_name):
        os.makedirs('storage/plots', exist_ok=True)

        grouped_results = defaultdict(list)
        for params, metrics in self.results:
            if metric_name not in metrics:
                print(f"Metric '{metric_name}' not found in results. Skipping.")
                continue
            grouped_results[params['num_topics']].append((params['topwords'], metrics[metric_name]))

        highest_scores = {
            num_topics: max(scores, key=lambda x: x[1])
            for num_topics, scores in grouped_results.items()
        }

        top_topics = sorted(highest_scores.items(), key=lambda x: x[1][1], reverse=True)[:5]

        filtered_results = {num_topics: grouped_results[num_topics] for num_topics, _ in top_topics}

        plt.figure(figsize=(12, 8))
        for num_topics, results in filtered_results.items():
            results = sorted(results, key=lambda x: x[0])
            x = [topwords for topwords, _ in results]
            y = [score for _, score in results]

            plt.plot(x, y, marker='o', label=f"Topics: {num_topics}")

        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Number of Top Words', fontsize=12)
        plt.ylabel(metric_name.capitalize(), fontsize=12)
        plt.title(f'Top 5 Topics by {metric_name.capitalize()} Scores vs Number of Top Words', fontsize=14)
        plt.legend(fontsize=10, title="Number of Topics")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        plot_path = os.path.join('storage/plots/fine_tuning', f"documents_{self.docs_number}_topwords_top_5_{metric_name}_results.png")
        plt.savefig(plot_path, dpi=300)
        plt.show()

        print(f"Plot saved as {plot_path}")


    def run(self, topics_range = range(5, 21), topwords_range = range(5, 21), metric_name='coherence', plot_results=True):
        for num_topics in topics_range:
            for topwords in topwords_range:
                params = {'num_topics': num_topics, 'topwords': topwords}
                print(f"Evaluating parameters: {params}")
                
                nmf_output = self.train_model(num_topics=num_topics, topwords=topwords)
                metrics = self.evaluate_model(nmf_output, topwords=topwords)
                self.results.append((params, metrics))
        
        best_params = self.get_best_params(metric_name)
        print(f"Best Parameters (based on {metric_name}): {best_params}")

        if plot_results:
            self.plot_results(metric_name)

        return best_params
