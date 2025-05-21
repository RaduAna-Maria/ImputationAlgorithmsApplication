import time
from sklearn.impute import SimpleImputer
from fancyimpute import IterativeImputer, SoftImpute

# Helper function to remove 'date' columns (or any datetime columns)
def remove_unwanted_columns(df, columns_to_keep):
    """
    Removes all columns that are not in the columns_to_keep list.

    :param df: DataFrame with columns to be filtered
    :param columns_to_keep: List of column names to keep in the DataFrame
    :return: DataFrame with only the columns specified in columns_to_keep
    """
    # Identify columns to keep that exist in the DataFrame
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    
    # Keep only the columns that are in the 'columns_to_keep' list
    df = df[columns_to_keep]
    
    return df

# 1. **Simple Imputation**
def simple_imputation(df, columns, strategy='mean'):
    """
    Applies simple imputation to specific columns.

    :param df: DataFrame with missing values
    :param columns: List of columns to apply imputation on
    :param strategy: The imputation strategy (e.g., 'mean', 'median', 'most_frequent')
    :return: DataFrame with imputed values
    """
    # Remove date columns
    df = remove_unwanted_columns(df, columns)

    # Apply imputer only on the specified columns
    imputer = SimpleImputer(strategy=strategy)
    
    start_time = time.time()
    df[columns] = imputer.fit_transform(df[columns])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.5f} seconds")

    return df

# 2. **Multiple Imputation (Iterative Imputation)**
def multiple_imputation(df, columns):
    """
    Applies multiple imputation (iterative imputation) to specific columns.

    :param df: DataFrame with missing values
    :param columns: List of columns to apply imputation on
    :return: DataFrame with imputed values
    """
    # Remove date columns
    df = remove_unwanted_columns(df, columns)

    # Apply imputer only on the specified columns
    imputer = IterativeImputer(max_iter=10, random_state=42)

    start_time = time.time()
    df[columns] = imputer.fit_transform(df[columns])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.5f} seconds")

    return df

# 3. **Expectation-Maximization (EM) Algorithm**
def em_imputation(df, columns):
    """
    Applies Expectation-Maximization (EM) imputation to specific columns.

    :param df: DataFrame with missing values
    :param columns: List of columns to apply imputation on
    :return: DataFrame with imputed values
    """
    # Remove date columns
    df = remove_unwanted_columns(df, columns)

    # Apply imputer only on the specified columns
    imputer = SoftImpute()

    start_time = time.time()
    df[columns] = imputer.fit_transform(df[columns])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.5f} seconds")

    return df
