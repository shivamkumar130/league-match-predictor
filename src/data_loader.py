from sklearn.model_selection import train_test_split
import pandas as pd
import torch

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    X = data.drop('win', axis=1)
    y = data['win']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def standardize_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def convert_to_tensors(X_train, X_test, y_train, y_test):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor