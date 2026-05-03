import pandas as pd
from utils.config import config

# this class is for cleaning the dataset, such as handling missing values, converting data types, delete unused columns, handling outliers, and other data cleaning steps
class DataCleaner:
    def __init__(self, df, dataset_type="prabayar"):
        self.df = df
        self.dataset_type = dataset_type

    def handle_missing_values(self):
        print("\nHandling missing values...")
        print(self.df.isnull().sum())
        
        for col in self.df.columns:
            if self.df[col].dtype in ['float64', 'int64']:
                if config['data_preprocessing']['numeric_missing_strategy'] == 'median':
                    median_value = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_value)
                    print(f"Filled missing values in numeric column '{col}' with median: {median_value}")
                elif config['data_preprocessing']['numeric_missing_strategy'] == 'mean':
                    mean_value = self.df[col].mean()
                    self.df[col] = self.df[col].fillna(mean_value)
                    print(f"Filled missing values in numeric column '{col}' with mean: {mean_value}")
                else:
                    print(f"Numeric missing value strategy '{config['data_preprocessing']['numeric_missing_strategy']}' is not supported. No imputation applied to column '{col}'.")
            elif config['data_preprocessing']['categorical_missing_strategy'] == 'most_frequent' and self.df[col].dtype == 'object':
                if col in config['categorical_features']['prabayar'] or col in config['categorical_features']['pascabayar']:
                    mode_value = self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(mode_value)
                    print(f"Filled missing values in categorical column '{col}' with mode: {mode_value}")
                else:
                    print(f"Categorical missing value strategy '{config['data_preprocessing']['categorical_missing_strategy']}' is not supported or column '{col}' is not in the expected categorical features for dataset type '{self.dataset_type}'. No imputation applied to column '{col}'.")
            else:
                print(f"No missing value strategy applied to column '{col}' with dtype '{self.df[col].dtype}'.")        
        
    def convert_data_types(self):
        print("\nConverting data types...")
        numeric_columns = set(config['numeric_features'].get(self.dataset_type, []))
        numeric_columns.add(config['target'].get(self.dataset_type))

        for col in self.df.columns:
            if col in numeric_columns:
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.replace(r"[^\d.\-]", "", regex=True)
                    .replace("", pd.NA)
                )
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

                if self.df[col].isna().any():
                    median_value = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_value)
                    print(f"Converted '{col}' to numeric and filled invalid values with median: {median_value}")
                    
    def drop_unused_columns(self, unused_cols):
        print("\nDropping unused columns...")
        existing_cols = [col for col in unused_cols if col in self.df.columns]
        self.df.drop(columns=existing_cols, inplace=True)
        print(f"Dropped columns: {existing_cols}")

    def handle_outliers(self):
        if config['data_preprocessing']['handle_outliers'] and config['data_preprocessing']['outlier_method'] == 'iqr':
            for col in self.df.select_dtypes(include=['float64', 'int64']).columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                print(f"Column '{col}' has {len(outliers)} outliers.")
                self.df[col] = self.df[col].clip(lower_bound, upper_bound)
                print(f"Clipped column '{col}' to bounds: {lower_bound}, {upper_bound}")
        else:
            print("Outlier handling is disabled or method is not supported.")

    def clean_data(self, unused_cols):
        self.handle_missing_values()
        self.convert_data_types()
        self.drop_unused_columns(unused_cols)
        self.handle_outliers()
        print("\nData cleaning completed.")
