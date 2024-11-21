import time
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
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


def objective(trial, X, y):

    param = {
        "objective": "binary",  # Modify for your task
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-3, 1e-1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "max_bin": trial.suggest_int("max_bin", 64, 255),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-3, 1.0, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "early_stopping_round": 50,
        "verbosity": -1,
    }
        
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create LGBM model
    model = LGBMRegressor(**param)
    
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)], 
              eval_metric='rmse')
    
    # Predict on the validation set
    y_pred = model.predict(X_val, num_iteration=model.best_iteration_)
    
    # Calculate RMSE
    rmse = root_mean_squared_error(y_val, y_pred)
    return rmse
