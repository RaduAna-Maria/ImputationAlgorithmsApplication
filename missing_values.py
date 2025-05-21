import numpy as np
import pandas as pd
import random

def introduce_missing_values(df, target_columns, percent_range=(10, 15), seed=None, missing_type="random"):
    """
    Introduces missing values (NaN) in specified columns using either random or burst pattern.

    :param df: The input DataFrame
    :param target_columns: List of column names where missing values should be added
    :param percent_range: Tuple specifying percentage range (min, max) of values to remove
    :param seed: Random seed for reproducibility (optional)
    :param missing_type: Type of missingness: 'random' or 'burst'
    :return: A new DataFrame with missing values introduced
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    df_missing = df.copy()
    target_columns = [col for col in target_columns if col in df_missing.columns]

    if not target_columns:
        print("[Warning] No valid columns provided for introducing missing values.")
        return df_missing

    total_target_cells = df_missing[target_columns].size
    percent_to_remove = random.uniform(*percent_range)
    num_missing = int(total_target_cells * percent_to_remove / 100)

    print(f"Introducing '{missing_type}' missing values in columns {target_columns}")
    print(f"{percent_to_remove:.2f}% of values → {num_missing} cells")

    if missing_type == "random":
        # Introduce missing values at random locations
        for _ in range(num_missing):
            row = random.randint(0, df_missing.shape[0] - 1)
            col = random.choice(target_columns)
            df_missing.at[row, col] = np.nan

    elif missing_type == "burst":
        # Introduce missing values in bursts (consecutive rows for a column)
        # burst_size can be random or sent as a parameter
        # hardcoded to 5 for a fair comparison between algorithms
        burst_size = 5 # random.randint(3, 7)
        num_bursts = max(1, num_missing // burst_size)

        for _ in range(num_bursts):
            col = random.choice(target_columns)
            start_idx = random.randint(0, df_missing.shape[0] - burst_size)
            df_missing.loc[start_idx:start_idx + burst_size - 1, col] = np.nan

    else:
        print(f"[Warning] Unknown missing_type: '{missing_type}'. Using default 'random'.")
        return introduce_missing_values(df, target_columns, percent_range, seed, missing_type="random")

    return df_missing
