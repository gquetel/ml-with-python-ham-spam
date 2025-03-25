import logging
from pathlib import Path

import numpy as np
from settings import RANDOM_STATE, BASE_PATH

from src.models.model import Model

from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def plot_and_save_tree(clf: DecisionTreeClassifier, model_name: str, columns: list):
    """Save decision tree plot under.

    A png image is save under ./output/model_name/plot_tree.png.

    Args:
        clf (DecisionTreeClassifier): Classifier to plot.
        model_name (str): Model name, used to create a subfolder.
        columns (list): Feature names to include in plot.
    """
    # ham = 0, spam = 1
    Path(BASE_PATH + "/output/" + model_name).mkdir(exist_ok=True)
    # We manually increase figure size to prevent overlap of nodes on big trees.
    plt.figure(figsize=(40, 15))
    plot_tree(clf, feature_names=columns, class_names=["ham", "spam"], fontsize=5)
    fp_tree_plot = "".join([BASE_PATH, "/output/", model_name, "/plot_tree.png"])
    plt.savefig(fp_tree_plot, dpi=100)
    logger.info("Saved plot of %s at %s", model_name, fp_tree_plot)


class DecisionTree(Model):
    """A Decision Tree Model.

    This class is a wrapper around the sklearn.tree.DecisionTreeClassifier class.
    - max_depth controls the depth of trees and therefore the overfitting.
    """

    def __init__(
        self,
        name: str = "DecisionTree",
        max_depth: int | None = None,
        random_state: int = RANDOM_STATE,
        column_names: list | None = None,
    ):
        self._name = name
        self._model = DecisionTreeClassifier(
            criterion="gini", max_depth=max_depth, random_state=random_state
        )
        self._column_names = column_names

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the given data.

        Args:
            X (np.ndarray): Training vectors
            y (np.ndarray): Target values
        """
        self._model.fit(X, y)
        if self._column_names != None:
            plot_and_save_tree(self._model, self._name, self._column_names)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify samples X.

        Args:
            X (np.ndarray): Samples to predict.

        Returns:
            np.ndarray: Predictions
        """
        return self._model.predict(X)

    def predict_and_evaluate(self, X: np.ndarray, y: np.ndarray):
        predictions = self.predict(X)
        errors = np.sum(predictions != y)
        logger.info(f"Misclassification errors: {errors}")

        return predictions


class RandomForest(Model):
    """A Random Forest classifier model.

    This class is a wrapper around the sklearn.ensemble.RandomForestClassifier class.
    """

    def __init__(
        self,
        name: str = "RandomForest",
        max_depth: int | None = None,
        random_state: int = RANDOM_STATE,
    ):
        self._name = name
        self._model = RandomForestClassifier(
            criterion="gini", max_depth=max_depth, random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the given data.

        Args:
            X (np.ndarray): Training vectors
            y (np.ndarray): Target values
        """
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Classify samples X.

        Args:
            X (np.ndarray): Samples to predict.

        Returns:
            np.ndarray: Predictions
        """
        return self._model.predict(X)

    def predict_and_evaluate(self, X: np.ndarray, y: np.ndarray):
        predictions = self.predict(X)
        errors = np.sum(predictions != y)
        logger.info(f"Misclassification errors: {errors}")

        return predictions
