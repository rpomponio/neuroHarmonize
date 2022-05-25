import pytest
import pickle
import os

def test_synthetic_data_exists():
    assert os.path.exists('synthetic_data.pkl')

def test_synthetic_data_shape():
    data = pickle.load(open('synthetic_data.pkl', 'rb'))

    assert data.shape == (200, 154)

if __name__ == "__main__":
    test_synthetic_data_exists()
    test_synthetic_data_shape()
    print("All tests passed")
