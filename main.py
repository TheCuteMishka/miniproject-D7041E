from evaluate.train import train_models
from data.prepare_datasets import get_data
from numpy import ndarray

def main():

    datasets: tuple[ndarray, ndarray, ndarray, ndarray, ndarray] = get_data()

    hyperparameters_to_vary: dict[str, tuple] = {
        "RandomForest": (50, 100, 150, 200, 250), # n-estimators
        "k-means": ("lloyd", "elkan") #TODO: Add hyperparameter to variate that is not FIXED!
    }

    trained_random_forest_models, trained_kmeans_models = train_models(datasets, hyperparameters_to_vary)

    print(
        trained_random_forest_models,
        "---------------------------",
        trained_kmeans_models,
        sep="\n"
          )

    # TODO: Make Testing calls below!


if __name__ == "__main__":
    main()
