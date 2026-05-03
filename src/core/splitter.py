import pandas as pd
from utils.config import config
import numpy as np

class Splitter:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df.copy()
        self.X = self.df.drop(columns=[target_column])
        self.y = self.df[target_column]

    def prepaid_split(self):
        # all columns except target = features
        # X is represented as a dataframe as a input, while y is represented as a target variable
        return self.X, self.y


    def postpaid_split(self):
        # all columns except target = features
        # X = self.df.drop(columns=[config['target']['pascabayar']])
        # target column
        # y = self.df[config['target']['pascabayar']]

        return self.X, self.y

    def split_dataset(self, train_ratio=0.8, random_state=42, shuffle=True):
        """
        train_ratio = 0.8  -> 80% train, 20% test
        train_ratio = 0.7  -> 70% train, 30% test
        """

        n = len(self.X)

        # ----------------------------------------------
        # Buat index data
        # ----------------------------------------------
        indices = np.arange(n)

        # ----------------------------------------------
        # Acak data
        # ----------------------------------------------
        if shuffle:
            np.random.seed(random_state)
            np.random.shuffle(indices)

        # ----------------------------------------------
        # Hitung jumlah train
        # ----------------------------------------------
        train_size = int(n * train_ratio)

        # ----------------------------------------------
        # Pisahkan index
        # ----------------------------------------------
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]

        # ----------------------------------------------
        # Ambil data
        # ----------------------------------------------
        
        X_train = self.X.iloc[train_idx]
        X_test  = self.X.iloc[test_idx]

        y_train = self.y.iloc[train_idx]
        y_test  = self.y.iloc[test_idx]

        # ----------------------------------------------
        # Output info
        # ----------------------------------------------
        print("Jumlah total data :", n)
        print("Jumlah data train:", len(X_train))
        print("Jumlah data test :", len(X_test))

        return X_train, X_test, y_train, y_test       
    