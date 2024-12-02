import numpy as np
from sklearn import datasets

def prepare_iris():
    iris = datasets.load_iris()

    data = iris["data"]
    targets = iris["target"]

    train_data = np.vstack((data[10:50], data[50 + 10 : 100], data[100 + 10: 150]))
    train_labels = np.concatenate((targets[10:50], targets[50 + 10 : 100], targets[100 + 10: 150]))

    test_data = np.vstack((data[:10], data[50 : 50 + 10], data[100: 110]))
    test_labels = np.concatenate((targets[:10], targets[50 : 50 + 10], targets[100: 110]))
    return train_data, train_labels, test_data, test_labels