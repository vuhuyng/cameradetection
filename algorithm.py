from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def train_knn(faces, labels, n_neighbors=5):
    """
    Train a KNN classifier without scaling or PCA.

    Parameters:
    - faces: 2D numpy array of shape (n_samples, n_features) containing face data.
    - labels: List or 1D numpy array of shape (n_samples,) containing labels.
    - n_neighbors: Number of neighbors to use for KNN.

    Returns:
    - Trained KNN model.
    """
    if faces.ndim != 2:
        raise ValueError(
            "Face data should be 2-dimensional (samples x features)")

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto')

    # Train the KNN classifier
    knn.fit(faces, labels)

    return knn
