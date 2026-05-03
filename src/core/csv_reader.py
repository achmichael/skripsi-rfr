import pandas as pd
import csv

class CSVReader:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load_dataset(self):
        try:
            df = pd.read_csv(self.file_path)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

        print(f"Dataset loaded successfully with shape: {df.shape}")
        print(f"Columns in the dataset: {df.columns.tolist()}")        

        # check for missing column value
        print("\nChecking for missing columns...")
        print(df.isnull().sum())
        
        
        # check type data
        print("\nChecking data types...")        
        print(df.dtypes)

        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().astype(str).head(5).tolist()
                print(f"\nSample values from column '{col}': {sample}")
        
        
        # check rupiah format
        print("\nChecking for 'rupiah' format in 'cost' column...")
        
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_text = " ".join(df[col].dropna().astype(str).head(20))
                if 'rupiah' in sample_text.lower() or '.' in sample_text or ',' in sample_text:
                    print(f"Column '{col}' contains 'rupiah' format.")
                    break
        
        print("\nDataset preview:")
        print(df.head())
        
        
    def validate_csv_structure(self, file_path):
        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            rows = list(reader)
            if len(rows) == 0:
                raise ValueError("CSV file is empty.")
            
            header = rows[0]
            expected_cols = list(header)

            print(f"Expected cols: {expected_cols}")
            
            valid = True
            
            for i, row in enumerate(rows[1:], start=2):
                if len(row) != len(expected_cols):
                    print(f"Row {i} has an incorrect number of columns: {len(row)} instead of {len(expected_cols)}")
                    valid = False
            
            if valid:
                print("CSV structure is valid.")
                
            return valid
    
    
            