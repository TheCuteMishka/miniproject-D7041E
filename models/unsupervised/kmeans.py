from sklearn.cluster import KMeans

class Kmeans:

    def __init__(self, num_clusters, varying_hyperparameter):
        self.internal_model = KMeans(n_clusters = num_clusters, algorithm = varying_hyperparameter)

    def fit(self, X, Y):
        return self.internal_model.fit(X, Y)