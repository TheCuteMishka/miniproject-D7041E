from evaluate.train import train_models
from evaluate.test import test_models
from evaluate.produce_table import create_table
from data.prepare_datasets import get_data
from typing import Any


def main():
    datasets: tuple[Any, ...] = get_data()

    hyperparameters_to_vary: dict[str, tuple] = {
        #"RandomForest": (50, 100, 150, 200, 250, 10_000), # n-estimators
        "RandomForest": (50, 100, 150, 200, 250, 250),
        "k-means": ("lloyd", "elkan") # Algorithm to chose
    }

    trained_random_forest_models, trained_kmeans_models = train_models(datasets, hyperparameters_to_vary)

    print(
        trained_random_forest_models,
        "---------------------------",
        trained_kmeans_models,
        sep="\n"
          )

    # Test will get the classification report into '.txt' format into respective files.
    test_models(trained_random_forest_models, trained_kmeans_models, datasets)

    # Gets a nice format of the result per model.
    print(create_table())

if __name__ == "__main__":
    main()

