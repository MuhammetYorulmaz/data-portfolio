import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        return pd.get_dummies(df[column], prefix='category', dtype=int)
    

def correlation_heatmap(corr: pd.DataFrame):
    """
    Plots a heatmap to show correlations between features.

    Args:
        corr (pd.DataFrame): A correlation matrix, usually created with DataFrame.corr().
    
    """
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(25,25))
    cmap = sns.diverging_palette(200, 20, as_cmap=False)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .5})

    f.text(0.40, 0.90, 'Correlation Heatmap', fontweight='bold', fontfamily='serif', fontsize=18)

    plt.tight_layout()
    plt.show()


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