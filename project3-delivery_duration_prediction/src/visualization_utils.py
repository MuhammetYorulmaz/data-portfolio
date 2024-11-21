import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
