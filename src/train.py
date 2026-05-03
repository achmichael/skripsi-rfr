# this script are used to train both model prabayar and pascabayar separately based on args command line, including data loading, preprocessing, feature engineering, model training, and evaluation
import argparse
import pandas as pd
import numpy as np
from core.preprocessor import Preprocessor
from forest.random_forest_regressor import RandomForestRegressor
from utils.config import config
from utils.file_writer import FileWriter

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

def main(dataset_type):
    df = pd.read_csv(config['paths']['raw_data'][dataset_type])
    target_column = config['target'][dataset_type]
    preprocessing_config = config["data_preprocessing"]
    train_ratio = 1 - preprocessing_config["test_size"]

    train_df, test_df = split_raw_dataset(
        df,
        train_ratio=train_ratio,
        random_state=preprocessing_config["random_state"],
        shuffle=preprocessing_config["shuffle"],
    )

    preprocessor = Preprocessor(dataset_type)
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

    rf_config = config["random_forest"]
    rf = RandomForestRegressor(
        n_estimators=rf_config["n_estimators"],
        max_depth=rf_config["max_depth"],
        min_samples_split=rf_config["min_samples_split"],
        min_samples_leaf=rf_config["min_samples_leaf"],
        max_features=rf_config["max_features"],
        random_state=rf_config["random_state"],
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mse = ((y_test - y_pred) ** 2).mean()
    print(f"Mean Squared Error: {mse}")

    file_writer = FileWriter()
    file_writer.save_model(rf, dataset_type)
    file_writer.save_evaluation_results(mse, dataset_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for prabayar or pascabayar dataset")
    parser.add_argument("--dataset", choices=["prabayar", "pascabayar"], required=True)
    args = parser.parse_args()
    main(args.dataset)
