from sklearn.ensemble import RandomForestClassifier

class RandomForest(RandomForestClassifier):
    def __init__(self, *hyperparameters):
        super().__init__(*hyperparameters)