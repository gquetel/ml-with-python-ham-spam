from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    def __init__(self):
        self._name = "Model"

    @property
    def name(self):
        return self._name

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns the predicted label for the given inputs.

        Args:
            X (np.ndarray): Input data
        """
        pass

    @abstractmethod
    def predict_and_evaluate(self, X: np.ndarray, y: np.ndarray):
        """Predicts the labels for the given inputs and compute evaluation metrics or plots depending on the model.

        Args:
            X (np.ndarray): Features
            y (np.ndarray): Ground truth labels
        """
        pass
