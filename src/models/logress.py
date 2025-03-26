import logging
import numpy as np

from sklearn.linear_model import LogisticRegression as skLR
from sklearn.preprocessing import StandardScaler

from settings import RANDOM_STATE
from src.models.model import Model
from src.utils.visualize import plot_losses_by_epoch

logger = logging.getLogger(__name__)


class LogisticRegression(Model):
    """Logistic regression classifier."""

    def __init__(
        self,
        lr: float,
        epochs: int,
        name: str = "Logistic-Regression",
        standardize: bool = False,
        random_state: int = RANDOM_STATE,
    ):
        self.lr = lr
        self.epochs = epochs
        self.rs = random_state
        self._name = name
        self.use_feature_standardization = standardize
        self._standardization_params = None

        self._w = None
        self._b = None
        self._losses = []

    def _compute_standardization_params(self, X: np.ndarray):
        """Compute the mean and standard deviation for each feature in the dataset from the training data and save them to also be applied to the test data.

        Args:
            X (np.ndarray): Vectors of features

        Returns:
            np.ndarray: Mean and standard deviation for each feature
        """
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        self._standardization_params = (means, stds)

    def _standardize_features(self, X: np.ndarray) -> np.ndarray:
        """Standardize the features of the dataset using the mean and standard deviation computed from the training data.

        Args:
            X (np.ndarray): Vectors of features

        Returns:
            np.ndarray: Standardized features
        """
        means, stds = self._standardization_params
        return (X - means) / stds

    def fit(self, X: np.ndarray, y: np.ndarray):
        rgen = np.random.default_rng(self.rs)
        self._w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self._b = 0
        self._losses = []

        if self.use_feature_standardization:
            self._compute_standardization_params(X)
            X = self._standardize_features(X)

        for i in range(self.epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self._w += self.lr * 2 / X.shape[0] * np.dot(X.T, errors)
            self._b += self.lr * 2 / X.shape[0] * errors.sum()
            # Log-likelyhood
            loss = (
                -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            ) / X.shape[0]
            self._losses.append(loss)
            logger.info(f"Epoch: {i}, Loss: {loss}")

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self._w) + self._b

    def activation(self, X: np.ndarray) -> np.ndarray:
        """Compute logistic sigmoid activation"""
        return 1.0 / (1.0 + np.exp(-np.clip(X, -250, 250)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.use_feature_standardization:
            X = self._standardize_features(X)
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

    def predict_and_evaluate(self, X, y):
        """Predict the class for the given vectors and evaluate the model.

        Plot the loss value across epochs.

        Args:
            X (_type_): Vectors of features
            y (_type_): Target values
        """
        plot_losses_by_epoch(self._losses, self.name, "Log-likelihood loss")
        predictions = self.predict(X)
        errors = np.sum(predictions != y)
        logger.info(f"Misclassification errors: {errors}")
        return predictions


class LogisticRegressionSklearn(Model):
    """A wrapper class relying on the LogisticRegression class implemented by sklearn."""

    def __init__(
        self,
        epochs: int,
        C: float = 100,
        name: str = "Logistic-Regression-sklearn",
        random_state: int = RANDOM_STATE,
    ):
        self._name = name
        self._model = skLR(C=C, max_iter=epochs, random_state=random_state)
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
