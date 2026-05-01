from sklearn.cluster import KMeans

def train_kmeans(X, k=4):
    """
    Train K-Means clustering model.

    Parameters:
        X (array): Scaled feature data
        k (int): Number of clusters

    Returns:
        model (KMeans): Trained model
        labels (array): Cluster assignments
    """

    # Initialize K-Means model
    model = KMeans(n_clusters=k, random_state=42)

    # Fit model and assign clusters
    labels = model.fit_predict(X)

    return model, labels


def compute_elbow(X):
    """
    Compute inertia values for different K values.

    Used to determine optimal number of clusters (Elbow Method).

    Returns:
        K_range (range)
        inertia (list)
    """

    inertia = []
    K_range = range(1, 11)

    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)

        # Inertia = sum of squared distances to centroids
        inertia.append(model.inertia_)

    return K_range, inertia