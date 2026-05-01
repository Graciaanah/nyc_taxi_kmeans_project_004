import matplotlib.pyplot as plt

def plot_elbow(K_range, inertia):
    """
    Plot Elbow Method graph.

    Helps identify optimal number of clusters.
    """

    plt.figure()

    # Plot K vs inertia
    plt.plot(K_range, inertia, marker='o')

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal K")

    # Save plot
    plt.savefig("elbow_plot.png")

    plt.show()


def plot_clusters(X, labels):
    """
    Visualize clusters using first two features.

    Note: Only works for 2D visualization.
    """

    plt.figure()

    # Scatter plot with cluster colors
    plt.scatter(X[:, 0], X[:, 1], c=labels)

    plt.title("K-Means Clusters")

    # Save figure
    plt.savefig("clusters.png")

    plt.show()