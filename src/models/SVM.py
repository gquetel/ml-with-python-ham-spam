import logging
import numpy as np
from settings import RANDOM_STATE

from settings import RANDOM_STATE
from src.models.model import Model

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC as skSVC

logger = logging.getLogger(__name__)


class SVCLin(Model):
    """A SVC Model using a linear kernel.

    This class is a wrapper around the sklearn.svm.SVC class.
    - C parameter controls the regularization, the lowest value, the highest regulation.
    """

    def __init__(
        self,
        C: float,
        name: str = "SVC-lin-sklearn",
        random_state: int = RANDOM_STATE,
    ):
        self._name = name
        self._model = skSVC(
            C=C, kernel="linear", max_iter=-1, random_state=random_state
        )
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


class SVCRBF(Model):
    """A SVC Model using a RBF kernel.

    - gamma is another hyperparameter influencing the decision boundary.
        Higher gamma values means a decision boundary closer to training samples.
    """

    def __init__(
        self,
        C: float,
        name: str = "SVC-rbf-sklearn",
        gamma: float | str = "scale",
        random_state: int = RANDOM_STATE,
    ):
        self._name = name
        self._model = skSVC(
            C=C, kernel="rbf", max_iter=-1, random_state=random_state, gamma=gamma
        )
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
