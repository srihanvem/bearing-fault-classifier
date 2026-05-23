import numpy as np
import pandas as pd


def simplify_fault_label(label: str) -> str:
    label = str(label)

    if "Ball" in label:
        return "ball"
    if "IR" in label:
        return "inner"
    if "OR" in label:
        return "outer"
    return "normal"


def load_and_prepare_data(filepath: str, test_size: float = 0.2, random_seed: int = 42):
    data = pd.read_csv(filepath)

    if "fault" not in data.columns:
        raise ValueError("Dataset must contain a 'fault' column.")

    data["label"] = data["fault"].apply(simplify_fault_label)
    data = data.drop(columns=["fault"])

    X = data.drop(columns=["label"]).values
    y, label_names = pd.factorize(data["label"])

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    X = (X - mean) / std

    np.random.seed(random_seed)
    indices = np.random.permutation(len(X))

    split_index = int((1 - test_size) * len(X))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices], list(label_names)