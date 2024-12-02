from sklearn.cluster import KMeans

class Kmeans:

    def __init__(self, varying_hyperparameter):
        self.internal_model = KMeans(n_clusters = 3, algorithm = varying_hyperparameter)

    def fit(self, X, Y):
        return self.internal_model.fit(X, Y)