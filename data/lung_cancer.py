from ucimlrepo import fetch_ucirepo
import numpy as np


def prepare_lungcancer():
    # fetch dataset
    lung_cancer = fetch_ucirepo(id=62)

    # Data and labels as numpy arrays.
    data = lung_cancer.data.features.to_numpy()
    labels = lung_cancer.data.targets.to_numpy().reshape(-1)

    missing_values_indices = []

    for y_idx in range(data.shape[0]):
        for x_idx in range(data.shape[1]):
            if np.isnan(data[y_idx][x_idx]):
                missing_values_indices.append((x_idx, y_idx))

    cached_cols_missing_values = {y: None for index, (x, y) in enumerate(missing_values_indices)}

    for index, (x, y) in enumerate(missing_values_indices):
        values_in_row_x_with_missing_values = [value for value in data[:, x] if not np.isnan(value)]

        # if not cached then cache it!
        if not cached_cols_missing_values[y]:
            missing_value = np.round(np.mean(values_in_row_x_with_missing_values))
            data[y, x] = missing_value
            cached_cols_missing_values[y] = missing_value

        # Otherwise just assign directly.
        else:
            data[y, x] = cached_cols_missing_values[y]

    # Paranoid sanity check
    for _, (x, y) in enumerate(missing_values_indices):
        if np.isnan(data[y, x]):
            raise RuntimeError("Logic incorrect for handling missing values!")


    labels_to_amount = {
        1: 0,
        2: 0,
        3: 0
    }


    for label in labels:
        labels_to_amount[label] += 1

    label_ratio_over_datasets = {
        1: -1,
        2: -1,
        3: -1
    }

    total_labels = 32

    for label, label_amount in labels_to_amount.items():
        label_ratio_over_datasets[label] = float(label_amount)/float(total_labels)


    train_set_to_total_ratio = 0.8

    ones_limit_train = int(total_labels * train_set_to_total_ratio * label_ratio_over_datasets[1])
    twos_limit_train = int(total_labels * train_set_to_total_ratio * label_ratio_over_datasets[2])
    threes_limit_train = int(total_labels * train_set_to_total_ratio * label_ratio_over_datasets[3])

    ones_limit_test = labels_to_amount[1] - ones_limit_train
    twos_limit_test = labels_to_amount[2] - twos_limit_train
    threes_limit_test = labels_to_amount[3] - threes_limit_train

    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    ones_in_train = 0
    ones_in_test = 0
    twos_in_train = 0
    twos_in_test = 0
    threes_in_train = 0
    threes_in_test = 0

    for index, label in enumerate(labels):
        if label == 1 and ones_in_train < ones_limit_train:
            ones_in_train += 1
            train_data.append(data[index])
            train_labels.append(label)

        elif label == 1 and ones_in_test < ones_limit_test:
            ones_in_test += 1
            test_data.append(data[index])
            test_labels.append(label)

        elif label == 2 and twos_in_train < twos_limit_train:
            twos_in_train += 1
            train_data.append(data[index])
            train_labels.append(label)

        elif label == 2 and twos_in_test < twos_limit_test:
            twos_in_test += 1
            test_data.append(data[index])
            test_labels.append(label)

        elif label == 3 and threes_in_train < threes_limit_train:
            threes_in_train += 1
            train_data.append(data[index])
            train_labels.append(label)

        elif label == 3 and threes_in_test < threes_limit_test:
            threes_in_test += 1
            test_data.append(data[index])
            test_labels.append(label)

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels) - 1
    test_labels = np.array(test_labels) - 1

    return train_data, train_labels, test_data, test_labels, "Lung Cancer"