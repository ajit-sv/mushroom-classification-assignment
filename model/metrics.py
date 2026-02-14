import pandas as pd
import numpy as np

def encode_categorical(df):
    mappings = {}
    for col in df.columns:
        unique_vals = df[col].unique()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        df[col] = df[col].map(mapping)
        mappings[col] = mapping
    return df, mappings

def train_test_split(X, y, test_size=0.2):
    n = len(X)
    split = int(n * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]