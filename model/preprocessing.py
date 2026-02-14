import pandas as pd
import numpy as np


def encode_categorical(df, mappings=None):
    """
    Encode all columns of `df` to integer codes.

    If `mappings` is None, build a mapping for each column from sorted unique
    values and return (df_encoded, mappings).

    If `mappings` is provided, it should be a dict where keys are column names
    and values are {original_value: code}. The function will apply those
    mappings to `df`. Values not present in a provided mapping are assigned a
    new integer code equal to max_code+1 for that column.

    Returns (df_encoded, mappings_used)
    """
    if mappings is None:
        mappings = {}
        for col in df.columns:
            unique_vals = sorted(df[col].unique())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            df[col] = df[col].map(mapping).astype(int)
            mappings[col] = mapping
        return df, mappings

    # mappings provided: apply but keep mapping stable
    mappings_used = {}
    for col in df.columns:
        col_mapping = mappings.get(col)
        if col_mapping is None:
            # No mapping for this column in provided mappings: build from data
            unique_vals = sorted(df[col].unique())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            df[col] = df[col].map(mapping).astype(int)
            mappings_used[col] = mapping
            continue

        # Apply existing mapping; unseen values get next integer code
        max_code = max(col_mapping.values()) if len(col_mapping) > 0 else -1

        def map_val(v):
            if v in col_mapping:
                return col_mapping[v]
            # assign unseen to new code (do not mutate original mapping)
            return max_code + 1

        df[col] = df[col].map(map_val).astype(int)
        mappings_used[col] = col_mapping

    return df, mappings_used
