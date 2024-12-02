import numpy as np
from sklearn import datasets

def prepare_wine():
    wine = datasets.load_wine()
    data = wine["data"]
    targets = wine["target"]

    # 59 samples - label 1
    # 71 samples - label 2
    # 48 samples - label 3
    # ordered

    train_data = np.vstack((data[12:59], data[59 + 12: 59 + 71], data[59 + 71 + 12: ]))
    train_labels = np.concatenate((targets[12:59], targets[59 + 12: 59 + 71], targets[59 + 71 + 12: ]))

    test_data = np.vstack((data[:12], data[59: 59 + 12], data[59 + 71: 59 + 71 + 12]))
    test_labels = np.concatenate((targets[:12], targets[59: 59 + 12], targets[59 + 71: 59 + 71 + 12]))

    return train_data, train_labels, test_data, test_labels, "wine"