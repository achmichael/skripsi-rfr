import numpy as np
from utils.config import config

def transform_target(y, dataset_type):
    transform_name = config.get("target_transform", {}).get(dataset_type, "none")

    if transform_name == "log1p":
        return np.log1p(y)

    return y

def inverse_transform_target(y, dataset_type):
    transform_name = config.get("target_transform", {}).get(dataset_type, "none")

    if transform_name == "log1p":
        return np.expm1(y)

    return y
