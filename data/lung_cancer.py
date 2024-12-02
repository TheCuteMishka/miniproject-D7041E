from ucimlrepo import fetch_ucirepo
import numpy as np


def prepare_lungcancer():
    # fetch dataset
    lung_cancer = fetch_ucirepo(id=62)

    # Data and labels as numpy arrays.
    data = lung_cancer.data.features.to_numpy()
    labels = lung_cancer.data.targets.to_numpy()

    print(data)
    print(data.shape)

    missing_values_indices = []

    print(
        f"y axis length: {data.shape[0]}",
        f"x axis length: {data.shape[1]}",
        sep="\n"
    )


    for y_idx in range(data.shape[0]):
        for x_idx in range(data.shape[1]):
            if np.isnan(data[y_idx][x_idx]):
                missing_values_indices.append((x_idx, y_idx))

    print(missing_values_indices)

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


