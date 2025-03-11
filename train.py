import logging
import pandas as pd
import numpy as np
import sys

from src.data.make_dataset import HamSpamDataset

from src.models.perceptron import Perceptron, PerceptronSklearn
from src.models.adaline import Adaline, AdalineSGD
from src.models.logress import LogisticRegression

from src.models.model import Model
from settings import BASE_PATH

logger = logging.getLogger(__name__)


def init_logger():
    """Initialize logging components."""
    lstdout = logging.StreamHandler(sys.stdout)
    lstdof = logging.Formatter(" %(message)s")
    lstdout.setFormatter(lstdof)
    logging.basicConfig(level=logging.INFO, handlers=[lstdout])


def init_dataset():
    dataset = HamSpamDataset(f"{BASE_PATH}/data/spamham.csv")
    df_train, df_test = dataset.get_train_test_split()
    logger.info(
        "Dataset loaded, train shape: %s, test shape: %s",
        df_train.shape,
        df_test.shape,
    )
    return df_train, df_test


def eval_model(model: Model, df_train: pd.DataFrame, df_test: pd.DataFrame):
    """Train and evaluate a model on the given preprocessed datasets.

    Args:
        model (Model): Model instance to train and evaluate
        df_train (pd.DataFrame): Training dataset
        df_test (pd.DataFrame): Testing dataset
    """
    ldf_train = df_train.copy()
    ldf_test = df_test.copy()

    logger.info("Training model: %s", model.name)
    labels = ldf_train["label"]
    features = ldf_train.drop("label", axis=1)
    model.fit(features.values, labels.values)

    logger.info("Evaluating model: %s", model.name)
    labels_test = ldf_test["label"]
    features_test = ldf_test.drop("label", axis=1)
    model.predict_and_evaluate(features_test.values, labels_test.values)


def main():
    init_logger()
    df_train, df_test = init_dataset()

    models = [
        Perceptron(0.01, 20),
        Adaline(0.00005, 20),
        Adaline(
            0.1,
            20,
            standardize=True,
            name="Adaline-std",
        ),
        AdalineSGD(0.01, 20),
        PerceptronSklearn(0.01, 20),
        LogisticRegression(0.001, 20),
        LogisticRegression(0.1, 20, standardize=True, name="Logistic-Regression-std"),
    ]

    for model in models:
        eval_model(model, df_train, df_test)


if __name__ == "__main__":
    main()
