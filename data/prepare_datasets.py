import numpy as np
from iris import prepare_iris
from wine import prepare_wine

def get_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # TODO: The other datasets have to be prepared!
    dataset = tuple(prepare_iris(), prepare_wine())

    return dataset