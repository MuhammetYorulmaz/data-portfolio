import time
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, root_mean_squared_error


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
