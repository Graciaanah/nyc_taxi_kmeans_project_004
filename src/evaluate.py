from sklearn.metrics import silhouette_score

def compute_silhouette(X, labels):
    """
    Calculate silhouette score to evaluate clustering quality.

    Higher score = better-defined clusters.

    Returns:
        float: silhouette score
    """

    return silhouette_score(X, labels)