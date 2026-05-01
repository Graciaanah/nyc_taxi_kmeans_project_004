import unittest
import pandas as pd
from src.preprocess import clean_data

class TestPreprocess(unittest.TestCase):
    """
    Unit tests for preprocessing functions
    """

    def test_clean_data(self):
        """
        Test that invalid rows are removed correctly
        """

        df = pd.DataFrame({
            'trip_distance': [1, -1],
            'fare_amount': [10, -5],
            'passenger_count': [1, 2]
        })

        cleaned = clean_data(df)

        # Expect only valid row to remain
        self.assertEqual(len(cleaned), 1)