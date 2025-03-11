import logging
import numpy as np

from settings import RANDOM_STATE
from src.models.model import Model
from src.utils.visualize import plot_losses_by_epoch

logger = logging.getLogger(__name__)


class Adaline(Model):
    """Adaline model implementation with full batch gradient descent."""
    def __init__(
        self,
        lr: float,
        epochs: int,
        name : str = "Adaline",
        standardize: bool = False,
        random_state: int = RANDOM_STATE,
    ):
        self.lr = lr
        self.epochs = epochs
        self.rs = random_state
        self.use_feature_standardization = standardize
        self._name = name   
        self._w = None
        self._b = None
        self._losses = None
        self._standardization_params = None

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
            # delta weights = learning rate * negative gradient
            # gradient = 2/ X.shape[0] * np.dot(X.T, errors)
            self._w += self.lr * 2 / X.shape[0] * np.dot(X.T, errors)
            # delta bias = learning rate * negative gradient
            self._b += self.lr * 2 / X.shape[0] * errors.sum()
            loss = (errors**2).mean()
            self._losses.append(loss)
            logger.info(f"Epoch: {i}, Loss: {loss}")

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self._w) + self._b

    def activation(self, X: np.ndarray) -> np.ndarray:
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.use_feature_standardization:
            X = self._standardize_features(X)
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

    def predict_and_evaluate(self, X, y):
        """ Predict the class for the given vectors and evaluate the model. Plot the loss value across epochs.

        Args:
            X (_type_): Vectors of features
            y (_type_): Target values
        """
        plot_losses_by_epoch(self._losses, self.name, "Mean Squared Error")
        predictions = self.predict(X)
        errors = np.sum(predictions != y)
        logger.info(f"Misclassification errors: {errors}")


class AdalineSGD(Model):
    def __init__(
        self,
        lr: float,
        epochs: int,
        name : str = "AdalineSGD",
        random_state: int = RANDOM_STATE,
    ):
        self.lr = lr
        self.epochs = epochs
        self.rs = random_state
        self._name = name
        self._w = None
        self._b = None
        self._losses = None
        self._standardization_params = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        rgen = np.random.default_rng(self.rs)
        self._w = rgen.normal(loc=0, scale=0.01, size=X.shape[1])
        self._b = 0
        self._losses = []

        self._compute_standardization_params(X)
        X = self._standardize_features(X)

        for i in range(self.epochs):
            r_idx = rgen.permutation(len(X))
            losses = []
            for xi, target in zip(X[r_idx], y[r_idx]):
                losses.append(self._update_weights(xi, target))
            self._losses.append(np.mean(losses))
            logger.info(f"Epoch: {i}, Loss: {self._losses[-1]}")

    def _standardize_features(self, X: np.ndarray) -> np.ndarray:
        """Standardize the features of the dataset using the mean and standard deviation computed from the training data.

        Args:
            X (np.ndarray): Vectors of features

        Returns:
            np.ndarray: Standardized features
        """
        means, stds = self._standardization_params
        return (X - means) / stds

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

    def _update_weights(self, xi: np.ndarray, target: int):
        """Update the weights and bias of the model given a single training sample

        Args:
            xi (np.ndarray): _description_
            target (_type_): _description_
        """
        output = self.activation(self.net_input(xi))
        error = target - output
        self._w += self.lr * 2 * xi * error
        self._b += self.lr * 2 * error
        loss = error**2
        return loss

    def net_input(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self._w) + self._b

    def activation(self, X: np.ndarray) -> np.ndarray:
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_std = self._standardize_features(X)
        return np.where(self.activation(self.net_input(X_std)) >= 0.5, 1, 0)

    def predict_and_evaluate(self, X: np.ndarray, y: np.ndarray):
        """Predict the class for the given vectors and evaluate the model. Plot the loss value across epochs.

        Args:
            X (np.ndarray): Vectors of features
            y (np.ndarray): Target values
        """
        predictions = self.predict(X)
        errors = np.sum(predictions != y)
        logger.info(f"Misclassification errors: {errors}")
        plot_losses_by_epoch(self._losses, self.name, "Mean Squared Error")
