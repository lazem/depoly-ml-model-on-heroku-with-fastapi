import sys
import os
import pytest
import pandas as pd
import numpy as np

from model_training.ml.data import process_data
from model_training.ml.model import compute_model_metrics, inference
from model_training.train_model import read_split_data, fit_save_model
from joblib import load
import sklearn

model_out = "test_model"
# cat_features = ["measure", "level"]
cat_features = ["level"]
label = "goal"
num_points = 100
test_points = 10


# class model_training_test(unittest.TestCase):
def fixed_data_constructor():
    """
    Fixed data constructor utility for tests
    """
    levels = ["high", "medium", "low"]
    goals = ["accept", "reject"]
    train_data = pd.DataFrame(
        {"measure": np.linspace(1, 10, num_points),
         "level": np.random.choice(levels, size=num_points),
         "goal": np.random.choice(goals, size=num_points),
         }
    )
    test_data = pd.DataFrame(
        {"measure": np.linspace(1, 10, test_points),
         "level": np.random.choice(levels, size=test_points),
         "goal": np.random.choice(goals, size=test_points),
         }
    )
    return train_data, test_data


def test_process_data():
    train, test = fixed_data_constructor()
    train_X, train_y, encoder, lb = process_data(train, categorical_features=cat_features, label="goal", training=True)

    assert train_X.shape == (num_points, 4)
    assert train_y.shape == (num_points,)

    test_X, test_y, _, _ = process_data(test, categorical_features=cat_features, label="goal", training=False,
                                        encoder=encoder, lb=lb)
    assert test_X.shape == (test_points, 4)
    assert test_y.shape == (test_points,)


def test_fit_save_model():
    df, _ = fixed_data_constructor()

    fit_save_model(model_out, train=df, cat_features=cat_features, label=label)

    # check to see that the model was created
    assert os.path.exists(f"{model_out}_predictor_model.joblib")
    assert os.path.exists(f"{model_out}_predictor_encoder.joblib")
    assert os.path.exists(f"{model_out}_predictor_lb.joblib")

    # Check that the model file can be loaded properly
    # (by type checking that it is a sklearn linear regression estimator)
    loaded_model = load(f"{model_out}_predictor_model.joblib")
    assert isinstance(loaded_model, sklearn.tree.DecisionTreeClassifier)


def test_compute_model_metrics():
    _, df = fixed_data_constructor()
    loaded_model = load(f"{model_out}_predictor_model.joblib")
    encoder = load(f"{model_out}_predictor_encoder.joblib")
    lb = load(f"{model_out}_predictor_lb.joblib")
    X, y, _, _ = process_data(df, categorical_features=cat_features, label="goal", training=False, encoder=encoder,
                              lb=lb)
    preds = inference(loaded_model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision >= 0
    assert recall >= 0
    assert fbeta >= 0
