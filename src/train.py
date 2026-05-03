# this script are used to train both model prabayar and pascabayar separately based on args command line, including data loading, preprocessing, feature engineering, model training, and evaluation
import argparse
import pandas as pd
from models.prabayar import PrabayarModel
from models.pascabayar import PascabayarModel
from core.splitter import Splitter
from forest.random_forest_regressor import RandomForestRegressor
from utils.config import config
from utils.file_writer import FileWriter

def main(dataset_type):
    # Step 1: Load data
    df = pd.read_csv(config['paths']['raw_data'][dataset_type])
    
    # Step 2: Preprocess data
    if dataset_type == "prabayar":
        model = PrabayarModel(df)
    else:
        model = PascabayarModel(df)
    
    df_preprocessed = model.preprocess_data()
    
    if dataset_type == "prabayar":
        target_column = config['target']['prabayar']
    else:
        target_column = config['target']['pascabayar']

    # Step 3: Split data
    
    splitter = Splitter(df_preprocessed, target_column=target_column)

    if dataset_type == "prabayar":
        splitter.prepaid_split()
    else:
        splitter.postpaid_split()
    
    X_train, X_test, y_train, y_test = splitter.split_dataset()

    # Step 4: Train model
    rf = RandomForestRegressor(n_estimators=100, max_depth=10)
    rf.fit(X_train, y_train)

    # Step 5: Evaluate model
    y_pred = rf.predict(X_test)
    mse = ((y_test - y_pred) ** 2).mean()
    print(f"Mean Squared Error: {mse}")

    # Step 6: save model and evaluation results
    file_writer = FileWriter()
    file_writer.save_model(rf, dataset_type)
    file_writer.save_evaluation_results(mse, dataset_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for prabayar or pascabayar dataset")
    parser.add_argument("--dataset", choices=["prabayar", "pascabayar"], required=True)
    args = parser.parse_args()
    main(args.dataset)