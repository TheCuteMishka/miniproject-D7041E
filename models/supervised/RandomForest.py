from sklearn.ensemble import RandomForestClassifier

class RandomForest:

    def __init__(self, varying_hyperparameter):
        self.internal_model = RandomForestClassifier(n_estimators=varying_hyperparameter)

    def fit(self, X, Y):
        return self.internal_model.fit(X, Y)