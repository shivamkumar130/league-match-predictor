def load_csv(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def split_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y

def standardize_data(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def convert_to_tensor(data):
    import torch
    return torch.tensor(data, dtype=torch.float32)

def calculate_accuracy(predictions, labels):
    return (predictions == labels).float().mean().item()