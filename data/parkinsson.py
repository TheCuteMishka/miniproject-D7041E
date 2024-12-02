from ucimlrepo import fetch_ucirepo
import numpy as np

def prepare_parkinsson():
    # Magic fetch code
    parkinsons = fetch_ucirepo(id=174)

    # data (as pandas dataframes)
    data = parkinsons.data.features
    labels = parkinsons.data.targets

    data = data.to_numpy()
    labels = labels.to_numpy()

    zero_counter = 0
    one_counter = 0

    for index, label in enumerate(labels):
        if label == 1:
            one_counter += 1

        if label == 0:
            zero_counter += 1

    print(zero_counter,
          one_counter,
          f"1 - {100*one_counter/(one_counter + zero_counter)} %",
          f"0 - {100*zero_counter/(one_counter + zero_counter)} %",
    )

    train_samples_amount = int(0.8 * (zero_counter + one_counter))
    test_samples_amount = (zero_counter + one_counter) - train_samples_amount

    # Preserve class proportion for sets both train and test
    ones_limit_train = int(train_samples_amount*0.7538461538461539)
    zeros_limit_train = int(train_samples_amount*0.24615384615384617)

    ones_limit_test = int(test_samples_amount*0.7538461538461539) + 1
    zeros_limit_test = int(test_samples_amount*0.24615384615384617) + 1

    print(
        f"ones_limit_train: {ones_limit_train}",
        f"zeros_limit_train: {zeros_limit_train}",
        f"ones_limit_test: {ones_limit_test}",
        f"zeros_limit_test: {zeros_limit_test}",
        sep="\n"
    )

    amount_of_train_ones = 0
    amount_of_train_zeros = 0
    amount_of_test_ones = 0
    amount_of_test_zeros = 0

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for value, label in zip(data, labels):

        if label == 1 and amount_of_train_ones < ones_limit_train:
            amount_of_train_ones += 1
            train_data.append(value)
            train_labels.append(label)

        elif label == 0 and amount_of_train_zeros < zeros_limit_train:
            amount_of_train_zeros += 1
            train_data.append(value)
            train_labels.append(label)

        elif label == 1 and amount_of_test_ones < ones_limit_test:
            amount_of_test_ones += 1
            test_data.append(value)
            test_labels.append(label)

        elif label == 0 and amount_of_test_zeros < zeros_limit_test:
            amount_of_test_zeros += 1
            test_data.append(value)
            test_labels.append(label)

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return train_data, train_labels.reshape(-1), test_data, test_labels.reshape(-1), "parkinsson"