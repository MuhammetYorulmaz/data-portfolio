import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_redundant_pairs(df):
    """
    Identify redundant pairs in a correlation matrix.

    This function returns a set of column pairs from the correlation matrix
    that are redundant. These include all diagonal pairs (self-correlations)
    and pairs in the lower triangle, as they are duplicates of the upper triangle.

    Parameters:
    - df (pd.DataFrame): DataFrame for which the redundant correlation pairs are identified.

    Returns:
    - set: A set of tuples, each containing a pair of column names.
    """
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=5):
    """
    Get the top absolute correlations in a DataFrame.

    Computes the correlation matrix, removes redundant pairs (lower triangular
    and diagonal), sorts by absolute correlation value, and returns the top n pairs.

    Parameters:
    - df (pd.DataFrame): DataFrame from which correlations are calculated.
    - n (int): Number of top correlations to return. Default is 5.

    Returns:
    - pd.Series: A series of the top n absolute correlations, indexed by the pair of column names.
    """
    corr_matrix = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    corr_matrix = corr_matrix.drop(labels=labels_to_drop)  # Drop redundant pairs
    top_corr = corr_matrix.sort_values(ascending=False)[:n]
    return top_corr


def compute_vif(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) scores for selected features.

    This function computes the VIF score for each feature provided in the list,
    helping to identify features with potential multicollinearity issues. A higher
    VIF indicates a higher correlation with other features.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the dataset with the specified features.
    - features (list): List of column names for which to calculate the VIF scores.

    Returns:
    - pd.DataFrame: A DataFrame with two columns:
        - 'feature': Name of each feature.
        - 'VIF': Calculated VIF score for each feature, sorted in ascending order.
    """
    
    vif_data = pd.DataFrame({'feature': features})
    vif_data['VIF'] = [variance_inflation_factor(df[features].values, i) for i in range(len(features))]
    return vif_data.sort_values(by=['VIF']).reset_index(drop=True)
