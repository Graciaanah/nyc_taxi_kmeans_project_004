import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """
    Clean dataset by:
    - Converting columns to numeric
    - Removing missing values
    - Filtering invalid values

    Parameters:
        df (DataFrame)

    Returns:
        df (DataFrame): Cleaned dataset
    """

    # Convert important columns to numeric (invalid values → NaN)
    numeric_cols = ['trip_distance', 'fare_amount', 'passenger_count']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows with missing values
    df = df.dropna()

    # Remove invalid values (e.g., negative distance or fare)
    df = df[df['trip_distance'] > 0]
    df = df[df['fare_amount'] > 0]

    return df


def feature_engineering(df):
    """
    Create time-based features safely
    """

    # Convert datetime (auto-detect format)
    df['tpep_pickup_datetime'] = pd.to_datetime(
        df['tpep_pickup_datetime'],
        errors='coerce'
    )

    # Extract hour
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour

    #  Only drop rows where pickup_hour is missing
    df = df[df['pickup_hour'].notna()]

    return df

def select_features(df):
    """
    Select relevant numerical features for K-Means clustering.

    Returns:
        DataFrame containing only selected features
    """

    features = [
        'trip_distance',
        'fare_amount',
        'passenger_count',
        'pickup_hour'
    ]

    return df[features].copy()


def scale_features(df):
    """
    Standardize features using StandardScaler.

    This ensures all features contribute equally to distance calculations.

    Returns:
        scaled_data (array)
        scaler (fitted scaler object)
    """

    scaler = StandardScaler()

    # Fit scaler and transform data
    scaled = scaler.fit_transform(df)

    return scaled, scaler