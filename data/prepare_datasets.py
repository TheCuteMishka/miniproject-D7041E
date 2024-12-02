import numpy as np
from data.iris import prepare_iris
from data.wine import prepare_wine
from data.parkinsson import prepare_parkinsson

def get_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # TODO: The other datasets have to be prepared!
    dataset = tuple([prepare_iris(), prepare_wine(), prepare_parkinsson()])

    return dataset