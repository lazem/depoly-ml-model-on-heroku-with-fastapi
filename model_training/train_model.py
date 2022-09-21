# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import sys

from model_training.ml.data import process_data
from model_training.ml.model import train_model, inference, compute_model_metrics
import joblib
import sklearn

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"
TEST_SIZE = 0.20


def read_split_data(data_path):
    """
    Get data for training and testing
    Parameters
    ----------
    data_path: str
        Training data path
    Returns
    -------
    train_df , test_df
    """
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=TEST_SIZE)
    return train, test


def fit_save_model(model_filename, train=None, data_path=None, cat_features=None, label=LABEL):
    """
    Fit ml model with training data and save its artifact
    Parameters
    ----------
    model_filename: str
        path for model output
    train pd.DataFrame
        dataframe for training the model
    cat_features: list
        List containing the names of the categorical features
    label: str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    Returns
    -------
    """
    if cat_features is None:
        cat_features = CAT_FEATURES

    if train is None:
        train, test = read_split_data(data_path)
    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True)
    # Train and save a model.
    model = train_model(X_train, y_train)
    # Save artifact
    joblib.dump(model, f"{model_filename}_predictor_model.joblib")
    joblib.dump(encoder, f"{model_filename}_predictor_encoder.joblib")
    joblib.dump(lb, f"{model_filename}_predictor_lb.joblib")

