import logging
import numpy as np

from settings import RANDOM_STATE
from src.models.model import Model
from src.utils.visualize import plot_errors_by_epoch

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron as skPerceptron
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class Perceptron(Model):
    def __init__(
        self,
        lr: float,
        epochs: int,
        name: str = "Perceptron",
        random_state: int = RANDOM_STATE,
    ):
        self.lr = lr
        self.epochs = epochs
        self.rs = random_state
        self._name = name
        self._w = None
        self._b = None
        self._errors = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        rgen = np.random.default_rng(self.rs)
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


class PerceptronSklearn(Model):
    def __init__(
        self,
        lr: float,
        epochs: int,
        name: str = "Perceptron-sklearn",
        random_state: int = RANDOM_STATE,
    ):
        self._name = name
        self._model = skPerceptron(eta0=lr, max_iter=epochs, random_state=random_state)
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
        """Classify samples X.

        Args:
            X (np.ndarray): Samples to predict.

        Returns:
            np.ndarray: Predictions
        """
        X_std = self._scaler.transform(X)
        return self._model.predict(X_std)

    def predict_and_evaluate(self, X: np.ndarray, y: np.ndarray):
        y_preds = self.predict(X)
        errors = np.sum(y_preds != y)
        logger.info(f"Misclassification errors: {errors}")
        logger.info(f"Accuracy: {"%.3f" % accuracy_score(y,y_preds)}")
