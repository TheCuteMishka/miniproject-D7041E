from typing import Any
from data.iris import prepare_iris
from data.wine import prepare_wine
from data.parkinsson import prepare_parkinsson
from data.lung_cancer import prepare_lungcancer
from data.fertility import prepare_fertility

def get_data() -> tuple[Any, ...]:

    dataset = tuple([prepare_iris(), prepare_wine(), prepare_parkinsson(), prepare_lungcancer(), prepare_fertility()])

    return dataset