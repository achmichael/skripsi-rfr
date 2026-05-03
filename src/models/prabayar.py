from utils.config import config
from core.data_cleaner import DataCleaner
from core.feature_engineer import FeatureEngineer
from core.encoder import Encoder

class PrabayarModel:
    def __init__(self, df):
        self.df = df.copy()
        self.dataset_type = "prabayar"
    
    def preprocess_data(self):
        # Step 1: Data cleaning
        cleaner = DataCleaner(self.df, dataset_type=self.dataset_type)
        cleaner.handle_missing_values()
        cleaner.convert_data_types()
        cleaner.drop_unused_columns(config['unused_features']['prabayar'])
        cleaner.handle_outliers()

        self.df = cleaner.df
        # Step 2: Feature engineering
        engineer = FeatureEngineer(self.df, dataset_type=self.dataset_type)
        engineer.engineer_features()
        self.df = engineer.df
        # Step 3: Encoding categorical features
        encoder = Encoder(self.df)
        encoder.encode_features()
        self.df = encoder.df
        return self.df

    
