import pandas as pd
import numpy as np

def encode_categorical(df):
    mappings = {}
    for col in df.columns:
        unique_vals = sorted(df[col].unique())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        df[col] = df[col].map(mapping)
        mappings[col] = mapping
    return df, mappings
