import os

def create_table():

    sep = r"\\" if os.name == "nt" else "/"
    results_directory: str = rf"results{sep}"

    # (0,0) is empty
    accuracy_table = [
        [0.0 for _ in range(8)] for _ in range(6)
    ]

    first_row = 0
    offset = 6

    # Indicies to model_type
    idx_to_model = {
        **{i: f"RandomForest_{i * 50}" for i in range(1, 6)},
        **{index + offset: f"kmeans_{algorithm}" for index, algorithm in enumerate(("elkan", "llovd"))}
    }

    column_number_to_dataset = {
        **{index + 1: dataset for index, dataset in enumerate(["iris", "fertility", "Lung Cancer", "parkinsson", "wine"])}
    }

    for index in range(1, len(accuracy_table[first_row])):
        accuracy_table[first_row][index] = idx_to_model[index]
        if index <= len(column_number_to_dataset.keys()):
            accuracy_table[index][0] = column_number_to_dataset[index]

    index_before_class_specific_first_column = 6

    dataset_to_amount_of_classes = {
        dataset: amount for dataset, amount in zip(["iris", "fertility", "Lung Cancer", "parkinsson", "wine"], [3, 2, 3, 2, 3])
    }

    model_names = ['RandomForest_50', 'RandomForest_100', 'RandomForest_150', 'RandomForest_200', 'RandomForest_250', 'kmeans_elkan', 'kmeans_lloyd']

    for file_name in os.listdir(results_directory):
        with (open(f"{results_directory}{file_name}", "r") as file):
            lines = file.readlines()

            column_index = [index + 1 for index, dataset in enumerate(["iris", "fertility", "Lung Cancer", "parkinsson", "wine"]) if dataset in file_name][0]

            row_number: int = 0

            for index, model_name in enumerate(model_names):
                if model_name in file_name:
                    row_number = index + 1

            current_dataset = [dataset for dataset in ["iris", "fertility", "Lung Cancer", "parkinsson", "wine"] if dataset in file_name][0]

            amount_of_classes_in_dataset = dataset_to_amount_of_classes[current_dataset]

            accuracy_row_index_in_file = index_before_class_specific_first_column + \
                                         amount_of_classes_in_dataset + \
                                         2

            accuracy = lines[accuracy_row_index_in_file].split()[1]

            accuracy_table[column_index][row_number] = f"{float(accuracy) * 100.0: 0.4f} %"

    max_chars_per_item = 20

    output_formatted = ""

    for index, line in enumerate(accuracy_table):

        for item in line:
            item = str(item)

            item += " ".join(["" for _ in range(max_chars_per_item - len(item))])

            output_formatted += item

        output_formatted += "\n"

    output_formatted = " "*3 + output_formatted[3:]

    return output_formatted

