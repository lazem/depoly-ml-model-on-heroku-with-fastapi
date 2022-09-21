from joblib import load
import logging
from model_training.ml.data import process_data
from model_training.ml.model import compute_model_metrics
from train_model import CAT_FEATURES, LABEL


def load_model_artifacts(model_name):
    loaded_model = load(f"{model_name}_predictor_model.joblib")
    encoder = load(f"{model_name}_predictor_encoder.joblib")
    lb = load(f"{model_name}_predictor_lb.joblib")
    return loaded_model, encoder, lb


def evaluate_per_slice(model_name, test_data):
    loaded_model, encoder, lb = load_model_artifacts(model_name)
    cat_features_metrics, cat_features = [], []
    for cat in CAT_FEATURES:
        for feature in test_data[cat].unique():
            cat_df = test_data[test_data[cat] == feature]

            x_test, y_test, _, _ = process_data(
                cat_df,
                categorical_features=CAT_FEATURES, training=False,
                label=LABEL, encoder=encoder, lb=lb)
            y_pred = loaded_model.predict(x_test)

            metrics = compute_model_metrics(y_test, y_pred)
            cat_features_metrics.append(metrics)

            cat_features.append(f"{cat}_{feature}")

    # with open(f'{root_path}/model/slice_output.txt', 'w') as file:
    with open('slice_output.txt', 'w') as so_file:
        for cat_features_metric in zip(cat_features, cat_features_metrics):
            logging.info(cat_features_metric)
            so_file.write(str(cat_features_metric))
