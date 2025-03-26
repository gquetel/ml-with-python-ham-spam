import logging
from pathlib import Path

import numpy as np
from settings import RANDOM_STATE, BASE_PATH

from src.models.model import Model
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier


logger = logging.getLogger(__name__)


class KNN(Model):
    """A K-NN Classifier Model.

    This class is a wrapper around the sklearn.neighbors.KNeighborsClassifier class.

    """

    def __init__(
        self,
        name: str = "5-NN",
        n_neighbors: int = 5,
        metric: str = "minkowski",
    ):
        self._name = name
        self._model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
        self._scaler = StandardScaler()


    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the given data.
        Before fitting the model, the features are standardized using the  StandardScaler class from sklearn.

        Args:
            X (np.ndarray): Training vectors
            y (np.ndarray): Target values
        """
        X_std = self._scaler.fit_transform(X)
        self._model.fit(X_std, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_std = self._scaler.transform(X)
        return self._model.predict(X_std)

    def predict_and_evaluate(self, X: np.ndarray, y: np.ndarray):
        predictions = self.predict(X)
        errors = np.sum(predictions != y)
        logger.info(f"Misclassification errors: {errors}")
        return predictions
