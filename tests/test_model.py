import unittest
import numpy as np
from src.model import train_kmeans

class TestModel(unittest.TestCase):

    def test_kmeans(self):
        X = np.array([[1, 2], [1, 4], [10, 10]])

        model, labels = train_kmeans(X, k=2)

        self.assertEqual(len(labels), 3)