import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from src.utils.config import config

class Encoder:
    def __init__(self, df, dataset_type="prabayar"):
        self.df = df.copy()
        self.dataset_type = dataset_type

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
        categorical_cols = config['categorical_features'][self.dataset_type]
        existing_cols = [
            col for col in categorical_cols if col in self.df.columns
        ]
        self.df = pd.get_dummies(self.df, columns=existing_cols, drop_first=False, dtype=int)
        print(f"Encoded columns {existing_cols} using one-hot encoding.")
                    
    def validate_numeric_features(self):
        non_numeric_cols = self.df.select_dtypes(include=["object", "string"]).columns
        if len(non_numeric_cols) > 0:
            raise ValueError(f"Warning: Non-numeric columns remain after encoding: {list(non_numeric_cols)}")
        else:
            print("All features are numeric after encoding.")

    def encode_features(self):
        self.encode_categorical_features_one_hot()
        self.validate_numeric_features()
