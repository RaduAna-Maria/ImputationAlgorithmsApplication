import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def load_data(file_path):
    """
    Loads a CSV file and converts the 'date' column to datetime.
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def show_basic_info(df):
    """
    Displays basic info and statistics about the DataFrame.
    """
    print("\n--- Basic Info ---")
    print(df.info())
    print("\n--- Summary Statistics ---")
    print(df.describe())

def check_missing_values(df):
    """
    Prints the count of missing values per column.
    """
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

def plot_distributions(df, columns):
    """
    Plots the distributions of specified columns in a single 2x2 grid figure.

    :param df: pandas DataFrame
    :param columns: List of column names to plot
    """
    n_cols = len(columns)
    rows, cols = 2, 2  # fixed 2x2 grid
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True)
        else:
            axes[i].text(0.5, 0.5, f"'{col}' not found", ha='center', va='center')
            axes[i].set_title(f"{col} (missing)")

    # Hide any unused subplots
    for j in range(len(columns), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_time_series(df, columns):
    """
    Plots time series data for given numeric columns in a 2x2 subplot grid.
    """
    if 'date' not in df.columns:
        raise ValueError("The dataset must contain a 'date' column for plotting.")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    selected_columns = [col for col in columns if col in df.columns][:4]
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(selected_columns):
        axes[i].plot(df['date'], df[col], label=col)
        axes[i].set_title(col)
        axes[i].set_xlabel("Date")
        axes[i].set_ylabel("Value")
        axes[i].grid(True)
        axes[i].legend()

    for j in range(len(selected_columns), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, columns):
    """
    Plots a heatmap showing correlations between specified columns.
    """
    existing_columns = [col for col in columns if col in df.columns]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[existing_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, columns):
    """
    Creates boxplots for the specified columns.
    """
    existing_columns = [col for col in columns if col in df.columns]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[existing_columns])
    plt.title("Boxplots")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def compute_mae(original_df, imputed_df, columns):
    """
    Computes and displays the Mean Absolute Error (MAE) between original and imputed DataFrames for specified columns.
    """
    print("\n--- Mean Absolute Error (MAE) ---")
    mae_scores = {}
    for col in columns:
        if col in original_df.columns and col in imputed_df.columns:
            mask = original_df[col].notna() & imputed_df[col].notna()
            score = mean_absolute_error(original_df.loc[mask, col], imputed_df.loc[mask, col])
            mae_scores[col] = score
            print(f"{col}: {score:.4f}")
        else:
            print(f"[Warning] Column '{col}' not found in both DataFrames.")
    return mae_scores

def compute_rmse(original_df, imputed_df, columns):
    """
    Computes and displays the Root Mean Squared Error (RMSE) between original and imputed DataFrames for specified columns.
    """
    print("\n--- Root Mean Squared Error (RMSE) ---")
    rmse_scores = {}
    for col in columns:
        if col in original_df.columns and col in imputed_df.columns:
            mask = original_df[col].notna() & imputed_df[col].notna()
            score = root_mean_squared_error(original_df.loc[mask, col], imputed_df.loc[mask, col])
            rmse_scores[col] = score
            print(f"{col}: {score:.4f}")
        else:
            print(f"[Warning] Column '{col}' not found in both DataFrames.")
    return rmse_scores

def apply_analitics(df, columns):
    show_basic_info(df)
    check_missing_values(df)
    plot_distributions(df, columns)
    plot_time_series(df, columns)
    # plot_correlation_matrix(df, columns)
    # plot_boxplots(df, columns)
