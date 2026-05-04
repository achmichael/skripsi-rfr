import os
import json
import datetime
import math
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

    def save_metric_bar(self, metrics, dataset_type):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{dataset_type}_{timestamp}.png"
        path = os.path.join(self.output_dir, filename)

        names = list(metrics.keys())
        values = [metrics[name] for name in names]
        plot_values = [value if math.isfinite(value) else 0 for value in values]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names, plot_values, color="#4C78A8")
        ax.set_title(f"Evaluation Metrics - {dataset_type}")
        ax.set_xlabel("Value (symlog scale)")
        ax.set_ylabel("Metric")
        ax.set_xscale("symlog")
        ax.grid(axis="x", linestyle="--", alpha=0.35)

        fig.text(
            0.5,
            0.01,
            "Keterangan: MAE, MSE, dan RMSE semakin kecil semakin baik. R2 semakin mendekati 1 semakin baik. MAPE dalam persen.",
            ha="center",
            fontsize=9,
        )

        for bar, value in zip(bars, values):
            label = f"{value:.4f}" if math.isfinite(value) else str(value)
            x_offset = 5 if bar.get_width() >= 0 else -5
            horizontal_alignment = "left" if bar.get_width() >= 0 else "right"
            ax.annotate(
                label,
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(x_offset, 0),
                textcoords="offset points",
                ha=horizontal_alignment,
                va="center",
                fontsize=9,
            )

        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print("Metric chart saved:", path)

    def save_model(self, model, model_name):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        path = os.path.join(self.output_dir, filename)

        with open(path, "wb") as f:
            import pickle
            pickle.dump(model, f)

        print("Model saved:", path)
    
    def save_evaluation_results(self, metrics, dataset_type):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{dataset_type}_{timestamp}.json"
        path = os.path.join(self.output_dir, filename)

        results = {
            "dataset": dataset_type,
            "metrics": metrics,
            "timestamp": timestamp
        }

        with open(path, "w") as f:
            json.dump(results, f, indent=4)

        print("Evaluation results saved:", path)
