import numpy as np
from settings import RANDOM_STATE
from src.models.model import Model
import logging
from src.utils.visualize import plot_errors_by_epoch

logger = logging.getLogger(__name__)


class Perceptron(Model):
    def __init__(
        self, lr: float, epochs: int, random_state: int = RANDOM_STATE
    ):
        self.lr = lr
        self.epochs = epochs
        self.rs = random_state
        self._name = "Perceptron"
        self._w = None
        self._b = None
        self._errors = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        rgen = np.random.RandomState(self.rs)
        self._w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self._b = 0
        self._errors = []

        for i in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self._w += update * xi
                self._b += update
                errors += int(update != 0.0)
            self._errors.append(errors)
            logger.info(f"Epoch: {i}, Errors: {errors}")

        return self

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self._w) + self._b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def predict_and_evaluate(self, X: np.ndarray, y: np.ndarray):
        # First, plot training errors across epochs
        plot_errors_by_epoch(self._errors, self.name)

        # Then predict and compute metrics.
        predictions = self.predict(X)
        errors = np.sum(predictions != y)
        logger.info(f"Misclassification errors: {errors}")
