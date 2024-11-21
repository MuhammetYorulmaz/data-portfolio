import pandas as pd


def create_dummies(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Generates a DataFrame with one-hot encoded columns.

    Parameters:
    - df (DataFrame): The original DataFrame.
    - column (str): The column to one-hot encode.

    Returns:
    - DataFrame: A DataFrame containing the one-hot encoded columns.
    """
    
    if column != 'store_primary_category':
        return pd.get_dummies(df[column], prefix=column, dtype=int)
        
    else:
        return pd.get_dummies(df[column], prefix='cuisine_category', dtype=int)


# Scaling function
def scale(scaler, X, y):
    """
    Scales features and target variable using the given scaler.

    Args:
    - scaler: Instance of sklearn scaler (e.g., MinMaxScaler, StandardScaler)
    - X: Features
    - y: Target variable

    Returns:
    - X_scaled: Scaled features
    - y_scaled: Scaled target variable
    - y_scaler: Scaler instance for the target variable
    """
    X_scaler = scaler
    y_scaler = scaler

    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))
    
    return X_scaled, y_scaled, y_scaler
