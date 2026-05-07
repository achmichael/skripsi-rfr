# this script are used to train both model prabayar and pascabayar separately based on args command line, including data loading, preprocessing, feature engineering, model training, and evaluation
import argparse
import pandas as pd
import numpy as np
from core.preprocessor import Preprocessor
from forest.random_forest_regressor import RandomForestRegressor
from utils.config import config
from utils.file_writer import FileWriter
from core.metrics import calculate_mean_baseline_metrics, calculate_metrics
from utils.core import transform_target, inverse_transform_target

def calculate_mean_baseline_metrics(y_train, y_test):
    baseline_prediction = np.full(len(y_test), np.mean(y_train))
    return calculate_metrics(y_test, baseline_prediction)

def print_metrics(title, metrics):
    print(title)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

def print_top_feature_importances(model, top_n=15):
    if not hasattr(model, "feature_importances_") or len(model.feature_importances_) == 0:
        return

    feature_importances = sorted(
        zip(model.feature_names_, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    print(f"Top {top_n} Feature Importances:")
    for feature_name, importance in feature_importances[:top_n]:
        print(f"{feature_name}: {float(importance)}")


def split_raw_dataset(df, train_ratio=0.8, random_state=42, shuffle=True):
    indices = np.arange(len(df))

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    train_size = int(len(df) * train_ratio)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print("Jumlah total data :", len(df))
    print("Jumlah data train:", len(train_df))
    print("Jumlah data test :", len(test_df))

    return train_df, test_df

def build_model():
    rf_config = config["random_forest"]
    return RandomForestRegressor(
        n_estimators=rf_config["n_estimators"],
        max_depth=rf_config["max_depth"],
        min_samples_split=rf_config["min_samples_split"],
        min_samples_leaf=rf_config["min_samples_leaf"],
        max_features=rf_config["max_features"],
        random_state=rf_config["random_state"],
    )

def cross_validate_raw_dataset(df, dataset_type, n_splits=5):
    preprocessing_config = config["data_preprocessing"]
    target_column = config["target"][dataset_type]
    indices = np.arange(len(df))

    if preprocessing_config["shuffle"]:
        rng = np.random.default_rng(preprocessing_config["random_state"])
        rng.shuffle(indices)

    folds = np.array_split(indices, n_splits)
    fold_metrics = []

    for fold_number, validation_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(indices, validation_idx, assume_unique=True)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        validation_df = df.iloc[validation_idx].reset_index(drop=True)

        preprocessor = Preprocessor(dataset_type)
        train_processed = preprocessor.fit_transform(train_df)
        validation_processed = preprocessor.transform(validation_df)

        X_train = train_processed.drop(columns=[target_column])
        y_train = train_processed[target_column]
        X_validation = validation_processed.drop(columns=[target_column])
        y_validation = validation_processed[target_column]

        model = build_model()
        model.fit(X_train, transform_target(y_train, dataset_type))
        y_pred = inverse_transform_target(model.predict(X_validation), dataset_type)
        metrics = calculate_metrics(y_validation, y_pred)
        fold_metrics.append(metrics)

        print(
            f"Fold {fold_number}/{n_splits} - "
            f"MAE: {metrics['MAE']:.4f}, "
            f"RMSE: {metrics['RMSE']:.4f}, "
            f"R2: {metrics['R2']:.4f}, "
            f"MAPE: {metrics['MAPE']:.4f}"
        )

    cv_metrics = {
        metric_name: float(np.mean([metrics[metric_name] for metrics in fold_metrics]))
        for metric_name in fold_metrics[0]
    }

    baseline_metrics = {
        metric_name: float(np.mean([
            calculate_mean_baseline_metrics(
                df.iloc[np.setdiff1d(indices, fold, assume_unique=True)][target_column],
                df.iloc[fold][target_column],
            )[metric_name]
            for fold in folds
        ]))
        for metric_name in fold_metrics[0]
    }

    print_metrics("Cross Validation Average Metrics:", cv_metrics)
    print_metrics("Cross Validation Mean Baseline Metrics:", baseline_metrics)

    return cv_metrics

def main(dataset_type):
    df = pd.read_csv(config['paths']['raw_data'][dataset_type])
    preprocessor = Preprocessor(dataset_type)
    target_column = config['target'][dataset_type]
    preprocessing_config = config["data_preprocessing"]
    train_ratio = 1 - preprocessing_config["test_size"]
    df = preprocessor.clean_raw_dataset(df, dataset_type)
    cv_config = config.get("cross_validation", {})
    if cv_config.get("enabled", False):
        cross_validate_raw_dataset(
            df,
            dataset_type,
            n_splits=cv_config.get("n_splits", 5),
        )

    train_df, test_df = split_raw_dataset(
        df,
        train_ratio=train_ratio,
        random_state=preprocessing_config["random_state"],
        shuffle=preprocessing_config["shuffle"],
    )

    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)

    X_train = train_processed.drop(columns=[target_column])
    y_train = train_processed[target_column]
    X_test = test_processed.drop(columns=[target_column])
    y_test = test_processed[target_column]

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and test feature columns are not aligned after preprocessing.")

    if target_column in X_train.columns or target_column in X_test.columns:
        raise ValueError("Target column leaked into model features.")

    rf = build_model()
    rf.fit(X_train, transform_target(y_train, dataset_type))

    y_pred = inverse_transform_target(rf.predict(X_test), dataset_type)
    evaluation_metrics = calculate_metrics(y_test, y_pred)
    baseline_metrics = calculate_mean_baseline_metrics(y_train, y_test)

    print_metrics("Evaluation Metrics:", evaluation_metrics)
    print_metrics("Mean Baseline Metrics:", baseline_metrics)
    print(f"Feature count used by model: {X_train.shape[1]}")
    print_top_feature_importances(rf)

    file_writer = FileWriter()
    file_writer.save_model(rf, dataset_type)
    file_writer.save_evaluation_results(evaluation_metrics, dataset_type)
    file_writer.save_feature_log(
        dataset_type,
        X_train.columns,
        getattr(rf, "feature_importances_", None),
    )
    file_writer.save_metric_bar(evaluation_metrics, dataset_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for prabayar or pascabayar dataset")
    parser.add_argument("--dataset", choices=["prabayar", "pascabayar"], required=True)
    args = parser.parse_args()    
    main(args.dataset)
