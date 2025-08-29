# tests/test_preprocessing.py
import unittest
import sys
import os

# Add absolute path of scripts folder to sys.path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from preprocessing import preprocess_data

class TestPreprocessing(unittest.TestCase):
    def test_preprocess(self):
        (X_train, X_test, y_train, y_test), le_target = preprocess_data()
        self.assertGreater(X_train.shape[0], 0)
        self.assertGreater(len(set(y_train)), 1)

if __name__ == "__main__":
    unittest.main()
