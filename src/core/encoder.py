import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from utils.config import config

class Encoder:
    def __init__(self, df):
        self.df = df.copy()

    def encode_categorical_features(self):
        print("\nEncoding categorical features...")
        for col in self.df.columns:
            if is_object_dtype(self.df[col]) or is_string_dtype(self.df[col]):
                if col in config['categorical_features']['prabayar'] or col in config['categorical_features']['pascabayar']:
                    self.df[col] = self.df[col].astype('category').cat.codes
                    print(f"Encoded column '{col}' using label encoding.")
                else:
                    print(f"Column '{col}' is not in the expected categorical features list. No encoding applied.")
    
    def encode_categorical_features_one_hot(self):
        print("\nEncoding categorical features using one-hot encoding...")
        for col in self.df.columns:
            if is_object_dtype(self.df[col]) or is_string_dtype(self.df[col]):
                if col in config['categorical_features']['prabayar'] or col in config['categorical_features']['pascabayar']:
                    dummies = pd.get_dummies(self.df[col], prefix=col)
                    self.df = pd.concat([self.df, dummies], axis=1)
                    self.df.drop(columns=[col], inplace=True)
                    print(f"Encoded column '{col}' using one-hot encoding.")
                else:
                    print(f"Column '{col}' is not in the expected categorical features list. No encoding applied.")
                    
    def encode_features(self):
        self.encode_categorical_features()
        self.encode_categorical_features_one_hot()
