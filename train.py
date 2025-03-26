import logging
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import plotly.graph_objects as go
from sklearn import metrics
from src.data.make_dataset import HamSpamDataset

from src.models.perceptron import Perceptron, PerceptronSklearn
from src.models.adaline import Adaline, AdalineSGD
from src.models.logress import LogisticRegression, LogisticRegressionSklearn
from src.models.SVM import SVCLin, SVCRBF
from src.models.trees import DecisionTree, RandomForest
from src.models.knn import KNN

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


def eval_model(
    model: Model, df_train: pd.DataFrame, df_test: pd.DataFrame
) -> np.ndarray:
    """Train and evaluate a model on the given preprocessed datasets.

    Args:
        model (Model): Model instance to train and evaluate
        df_train (pd.DataFrame): Training dataset
        df_test (pd.DataFrame): Testing dataset
    """
    ldf_train = df_train.copy()
    ldf_test = df_test.copy()

    logger.info("Training and testing model: %s", model.name)
    labels = ldf_train["label"]
    features = ldf_train.drop("label", axis=1)
    model.fit(features.values, labels.values)

    labels_test = ldf_test["label"]
    features_test = ldf_test.drop("label", axis=1)
    return model.predict_and_evaluate(features_test.values, labels_test.values)


def save_scores_as_plots(
    lmodels: list, lf1scores: list, laccscores: list, lrecallscores: list
):
    folder_path = "./img/"
    Path(folder_path).mkdir(exist_ok=True)

    fig = go.Figure(
        data=[
            go.Bar(name="F1 Score", x=lmodels, y=lf1scores),
            go.Bar(name="Accuracy Score", x=lmodels, y=laccscores),
            go.Bar(name="Recall Score", x=lmodels, y=lrecallscores),
        ]
    )
    fig.update_layout(
        autosize=False,
        width=1000, # Increment value when more models are added.
        height=500,
        barmode="group",
        template="seaborn",
    )
    fig.write_image(folder_path + "scores.png")


def main():
    init_logger()
    df_train, df_test = init_dataset()

    lr = 0.0001
    lr_std = 0.1  # Feature standardization allows higher convergence therefore we can use a higher LR value.
    feature_names = list(df_train.columns)
    epochs = 50
    models = [
        Perceptron(lr, epochs),
        Adaline(lr, epochs),
        Adaline(
            lr_std,
            epochs,
            standardize=True,
            name="Adaline-std",
        ),
        AdalineSGD(lr, epochs),
        PerceptronSklearn(lr, epochs),
        LogisticRegression(0.005, epochs),
        LogisticRegression(
            lr_std, epochs, standardize=True, name="Logistic-Regression-std"
        ),
        LogisticRegressionSklearn(epochs, C=100),
        SVCLin(1.0),
        SVCRBF(10.0),
        DecisionTree(column_names=feature_names),
        DecisionTree(
            max_depth=4, name="DecisionTree-depth4", column_names=feature_names
        ),
        RandomForest(),
        RandomForest(max_depth=4, name="RandomForest-depth4"),
        KNN(),
        KNN(metric="cosine",name="5-NN-cosine"),
        KNN(name="3-NN-cosine", n_neighbors=3)

    ]

    targets = df_test["label"]
    lmodels = []
    lf1scores = []
    laccscores = []
    lrecallscores = []

    for model in models:
        preds = eval_model(model, df_train, df_test)
        lmodels.append(model.name)
        lf1scores.append(metrics.f1_score(targets, preds))
        laccscores.append(metrics.accuracy_score(targets, preds))
        lrecallscores.append(metrics.recall_score(targets, preds))

    save_scores_as_plots(lmodels, lf1scores, laccscores, lrecallscores)


if __name__ == "__main__":
    main()
