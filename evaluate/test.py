from sklearn.metrics import classification_report
import os

# TODO: Implement test logic by getting the data with 'get_data()'
#       it returns a tuple of all datasets in the project.
def test_models(trained_random_forest_models, trained_kmeans_models, datasets):
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    name_to_dataset = {
        dataset_name: (test_data, test_labels)  for _, _, test_data, test_labels, dataset_name in datasets
    }

    sep = "\\" if os.name == "nt" else "/"

    for hyperparameter_name, random_forest_model, dataset_name in trained_random_forest_models:

        test_data, test_labels = name_to_dataset[dataset_name]

        predictions = random_forest_model.predict(test_data)

        report = classification_report(test_labels, predictions)

        report = "\n".join([
            "-" * 60,
            "",
            f"dataset: {dataset_name}\thyperparmeter: {hyperparameter_name}\tmodel: {random_forest_model}",
            "",
            "-" * 60,
            report,
            "-" * 60,
            "",
            "-" * 60
        ])

        with open(f"results{sep}RandomForest_{hyperparameter_name}_{dataset_name}_results.txt", "w") as file:
            file.write(report)

    for hyperparameter_name, kmeans_model, dataset_name in trained_kmeans_models:
        test_data, test_labels = name_to_dataset[dataset_name]

        predictions = kmeans_model.predict(test_data)

        report = classification_report(test_labels, predictions)

        report = "\n".join([
                "-" * 60,
                "",
                f"dataset: {dataset_name}\thyperparmeter: {hyperparameter_name}\tmodel: {kmeans_model}",
                "",
                "-" * 60,
                report,
                "-" * 60,
                "",
                "-" * 60
            ])

        with open(f"results{sep}kmeans_{hyperparameter_name}_{dataset_name}_results.txt", "w") as file:
            file.write(report)
