import numpy as np

class FeatureSampler:
    def __init__(self, max_features="sqrt", random_state=42):
        self.max_features = max_features
        self.random_state = random_state
    
    def sample_features(self, n_features):
        if self.max_features == "sqrt":
            n_sampled = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            n_sampled = int(np.log2(n_features))
        else:
            n_sampled = n_features
        np.random.seed(self.random_state)

        return np.random.choice(n_features, n_sampled, replace=False)
    