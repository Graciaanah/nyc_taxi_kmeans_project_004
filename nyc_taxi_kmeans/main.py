from src.data_loader import load_data
from src.preprocess import clean_data, feature_engineering, select_features, scale_features
from src.model import train_kmeans, compute_elbow
from src.evaluate import compute_silhouette
from src.visualize import plot_elbow, plot_clusters

def main():
    """
    Main pipeline for K-Means clustering project.

    Steps:
    1. Load data
    2. Clean and preprocess
    3. Select and scale features
    4. Determine optimal K
    5. Train model
    6. Evaluate and visualize results
    """

    # Load dataset
    df = load_data("data/raw/2018_Yellow_Taxi_Trip_Data_20260501.csv")
    #  SAMPLE DATA (IMPORTANT for speed)
    df = df.sample(10000, random_state=42)
    # Clean and preprocess data
    df = clean_data(df)
    df = feature_engineering(df)

    # Select features for clustering
    features = select_features(df)

    # Scale features
    X, scaler = scale_features(features)

    # Determine optimal K using Elbow Method
    K_range, inertia = compute_elbow(X)
    plot_elbow(K_range, inertia)

    # Train K-Means model (choose K based on elbow plot)
    model, labels = train_kmeans(X, k=4)

    # Evaluate clustering
    score = compute_silhouette(X, labels)
    print("Silhouette Score:", score)

    # Visualize clusters
    plot_clusters(X, labels)


if __name__ == "__main__":
    main()