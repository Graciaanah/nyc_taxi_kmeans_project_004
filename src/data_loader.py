import pandas as pd

def load_data(file_path):
    """
    Load dataset from a CSV file into a Pandas DataFrame.

    Parameters:
        file_path (str): Path to the CSV file

    Returns:
        df (DataFrame): Loaded dataset
    """

    # Read CSV file (low_memory=False avoids dtype warnings)
    df = pd.read_csv(file_path, low_memory=False)

    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()

    return df