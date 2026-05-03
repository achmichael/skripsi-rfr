import os
import json
import datetime
import matplotlib.pyplot as plt

class FileWriter:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_results(self, model_name, results):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.json"
        path = os.path.join(self.output_dir, filename)

        with open(path, "w") as f:
            json.dump(results, f, indent=4)

        print("Saved:", path)

    def plot_metric_bar(self, metrics):
        names = list(metrics.keys())
        values = list(metrics.values())

        plt.figure(figsize=(8,5))
        plt.bar(names, values)
        plt.title("Evaluation Metrics")
        plt.ylabel("Value")
        plt.show()

    def save_model(self, model, model_name):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        path = os.path.join(self.output_dir, filename)

        with open(path, "wb") as f:
            import pickle
            pickle.dump(model, f)

        print("Model saved:", path)
    
    def save_evaluation_results(self, mse, dataset_type):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{dataset_type}_{timestamp}.json"
        path = os.path.join(self.output_dir, filename)

        results = {
            "dataset": dataset_type,
            "mse": mse,
            "timestamp": timestamp
        }

        with open(path, "w") as f:
            json.dump(results, f, indent=4)

        print("Evaluation results saved:", path)