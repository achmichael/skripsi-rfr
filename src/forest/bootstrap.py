import numpy as np

class Bootstrap:
    def __init__(self, df, dataset_type="prabayar", random_state=None):
        self.df = df.copy()
        self.dataset_type = dataset_type.lower()
        self.rng = np.random.default_rng(random_state)

    def create_bootstrap_sample(self):
        positions = self.rng.choice(len(self.df), len(self.df), replace=True)
        return self.df.iloc[positions], positions
    
    def create_oob_sample(self, bootstrap_positions):
        mask = np.ones(len(self.df), dtype=bool)
        mask[bootstrap_positions] = False
        return self.df.iloc[mask]
