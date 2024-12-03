from ucimlrepo import fetch_ucirepo
import numpy as np


# Prepares the fertility dataset
def prepare_fertility():
    fertility = fetch_ucirepo(id=244)

    # data (as pandas dataframes)
    data = fertility.data.features.to_numpy()
    labels = fertility.data.targets.to_numpy()

    # 'N' = 0, 'O' = 1
    labels = np.array(list(0 if label == 'N' else 1 for label in labels))

    class_to_amount = {
        0: 0,
        1: 0
    }

    for label in labels:
        class_to_amount[label] += 1

    # Approximate split ratio integer constructor conversion can make it slightly of for example.
    train_data_ratio = 0.8

    train_data_samples_first_class = int(train_data_ratio * float(class_to_amount[0]))
    train_data_samples_second_class = int(train_data_ratio * float(class_to_amount[1]))

    test_data_samples_first_class = class_to_amount[0] - train_data_samples_first_class
    test_data_samples_second_class = class_to_amount[1] - train_data_samples_second_class

    train_data_first_samples = 0
    train_data_second_samples = 0
    test_data_first_samples = 0
    test_data_second_samples = 0

    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    for index, label in enumerate(labels):

        if label == 0 and train_data_first_samples < train_data_samples_first_class:
            train_data_first_samples += 1
            train_data.append(data[index])
            train_labels.append(label)

        elif label == 1 and train_data_second_samples < train_data_samples_second_class:
            train_data_second_samples += 1
            train_data.append(data[index])
            train_labels.append(label)

        elif label == 0 and test_data_first_samples < train_data_samples_first_class:
            test_data_first_samples += 1
            test_data.append(data[index])
            test_labels.append(label)

        elif label == 1 and test_data_second_samples < test_data_samples_second_class:
            test_data_second_samples += 1
            test_data.append(data[index])
            test_labels.append(label)

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels, "fertility"