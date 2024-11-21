import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import time 
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, root_mean_squared_error


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


def visualize_outlier(df):

    """
  Visualizes outliers in a DataFrame using boxplots for specific columns.

  Parameters:
  df (pd.DataFrame): The DataFrame containing the columns to plot. 
                      It should have the following columns:
                      - 'subtotal'
                      - 'min_item_price'
                      - 'max_item_price'
                      - 'total_delivery_duration'
    """
    green_diamond = dict(markerfacecolor='g', marker='D')

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.boxplot(df["subtotal"],flierprops=green_diamond)
    ax1.set_title("Subtotal Boxplot",
                fontdict=dict(
                    family="DejaVu Sans",
                    size=15,
                    color="sienna"))

    ax2.boxplot(df["min_item_price"],flierprops=green_diamond)
    ax2.set_title("Minimum Item Price Boxplot",
                fontdict=dict(
                    family="DejaVu Sans",
                    size=15,
                    color="sienna"))

    ax3.boxplot(df["max_item_price"],flierprops=green_diamond)
    ax3.set_title("Maximum Item Price Boxplot",
                fontdict=dict(
                    family="DejaVu Sans",
                    size=15,
                    color="sienna"))

    ax4.boxplot(df["total_delivery_duration"],flierprops=green_diamond)
    ax4.set_title("Total Delivery Duration Boxplot",
                fontdict=dict(
                    family="DejaVu Sans",
                    size=15,
                    color="sienna"))
    plt.show()

def plot_regression(df):
    """
    Creates a regression plot for 'total_items' vs 'total_delivery_duration' from the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing 'total_items' and 'total_delivery_duration' columns.

    """
    fig = plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 1, 1)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    
    sns.regplot(x='total_items', y='total_delivery_duration', data=df, color='#244747')
    sns.despine()
    
    plt.ylabel('Total Delivery Duration')
    
    plt.show()
    
    
def plot_gini_importance(data):
    """
    Creates a bar plot for feature importance based on Gini-Importance.

    Parameters:
    data (pd.DataFrame): A DataFrame containing at least two columns:
                         - 'Feature': The names of the features.
                         - 'Gini-Importance': The corresponding importance values.
    """
    
    plt.figure(figsize=(15, 8))
    
    sns.barplot(y='Gini-Importance', x='Feature', data=data)
    
    plt.title('Feature Importance (Gini)', fontsize=16)
    plt.xlabel('Gini-Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.xticks(rotation=90)
    
    plt.show()
    

# RMSE function for inverse scaling
def rmse_inverse(y_true, y_pred, y_scaler):
    y_true_orig = y_scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred_orig = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
    return root_mean_squared_error(y_true_orig, y_pred_orig)

# Regression function
def create_regression(X, y, model, model_name, scaler_name, y_scaler=None, verbose=True):
    """
    Trains a regression model using cross-validation and computes RMSE.
    Handles scaled and unscaled data correctly.

    Args:
    - X: Features (scaled or unscaled)
    - y: Target variable (scaled or unscaled)
    - model: Regression model to use
    - model_name: Name of the model
    - scaler_name: Name of the scaler used
    - y_scaler: Target scaler for inverse scaling RMSE (if applicable)
    - verbose: Whether to print detailed output

    Returns:
    - rmse_scores: List of RMSE scores across folds
    - elapsed_time: Total time taken for cross-validation
    """
    start_time = time.time()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    if scaler_name != 'Without Scale' and y_scaler is not None:
        rmse_scorer = make_scorer(lambda y_true, y_pred: rmse_inverse(y_true, y_pred, y_scaler), greater_is_better=False)
    else:
        rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
        
    rmse_scores = cross_val_score(model, X, y, cv=kf, scoring=rmse_scorer)
    end_time = time.time()
    elapsed_time = end_time - start_time

    if verbose:
        print(f'Average RMSE : {-np.mean(rmse_scores):.4f} - Time taken: {elapsed_time:.4f} seconds')

    return -np.mean(rmse_scores), round(elapsed_time, 4)


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