from sklearn.cluster import KMeans
import numpy as np
from models.supervised.RandomForest import RandomForest
from models.unsupervised.kmeans import Kmeans

def train_models(datasets, hyperparameters_to_vary: dict[str, tuple]):

    trained_random_forest_models = []
    trained_kmeans_models = []

    # (50, 100, 150, 200, 250) - trees
    random_forest_hyperparameters = hyperparameters_to_vary["RandomForest"]

    # ("lloyd", "elkan") - algorithms
    kmeans_hyperparameters = hyperparameters_to_vary["k-means"]

    for dataset in datasets:
        train_data, train_labels, test_data, test_labels, dataset_name = dataset
        print(
            "--- DATASET - SHAPES ---",
            f"Dataset = {dataset_name}",
            f"train_data: {train_data.shape}",
            f"train_labels: {train_labels.shape}",
            f"test_data: {test_data.shape}",
            f"test_labels: {test_labels.shape}",
            f"------------------------",
            "",
            sep="\n"
        )

        for random_forest_hyperparameter in random_forest_hyperparameters:
            # Creates the models with the default parameters but varies the amount of estimators
            random_forest_model = RandomForest(random_forest_hyperparameter)

            random_forest_model = random_forest_model.fit(train_data, train_labels)
            trained_random_forest_models.append((random_forest_hyperparameter, random_forest_model, dataset_name))

        # The parameter that can vary.
        for kmeans_hyperparameter in kmeans_hyperparameters:
            # Gets all unique values in the labels (classes) and calculates
            # in the returned numpy array how many elements there are.
            num_clusters = len(np.unique(train_labels))

            # Creates the model with the default parameters but varies the algorithm argument
            kmeans_model = Kmeans(num_clusters, kmeans_hyperparameter)

            kmeans_model = kmeans_model.fit(train_data, train_labels)

            trained_kmeans_models.append((kmeans_hyperparameter, kmeans_model, dataset_name))

    return trained_random_forest_models, trained_kmeans_models