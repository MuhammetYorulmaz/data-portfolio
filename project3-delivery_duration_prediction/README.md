# Delivery Duration Prediction

## Project Overview

This project aims to develop a machine learning model to predict the total time taken for a delivery, measured in seconds.

## Data Source

The dataset used for this project is sourced from [StrataScratch](https://platform.stratascratch.com/data-projects/delivery-duration-prediction).

## Requirements

To run this project, you need to set up the required environment and install dependencies. The project uses `conda` to manage the environment, and the required libraries are listed in the `environment.yml` file.

### Dependencies

- Python 3.x
- Conda or Miniconda to manage the environment
- Libraries: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `tensorflow`, `matplotlib`, and others as specified in the `environment.yml` file.

## Setup Instructions

### 1. Clone the repository

Start by cloning the project repository to your local machine:

```bash
git clone https://github.com/MuhammetYorulmaz/data-portfolio.git
cd data-portfolio/project3-delivery_duration_prediction
```

### 2. Create and activate the Conda environment

To set up the environment with all necessary dependencies, run the following commands:

```bash
conda env create -f environment.yml
conda activate delivery_duration_prediction_env
```
### 3. Open the project in Visual Studio Code (VSCode)

1. **Launch Visual Studio Code**: If you have Visual Studio Code installed, open it from your terminal or use the shortcut.

2. **Open the project folder**: In VSCode, click on **File** > **Open Folder** and select the project folder (`data-portfolio/project3-delivery_duration_prediction`).

3. **Install Python Extension (if not already installed)**: If you haven’t installed the Python extension for VSCode, you’ll be prompted to install it when you open a `.ipynb` file. This extension is required to work with Jupyter notebooks.

4. **Select the Conda environment in VSCode**:
    - After opening the project in VSCode, click on the **Python environment** in the bottom left corner.
    - Select the `delivery_duration_prediction_env` Conda environment from the list. This ensures that the environment with the required dependencies is used.

5. **Open the Jupyter notebook**: Open the file `delivery_duration_prediction.ipynb` by navigating to it in the file explorer on the left side of VSCode.

## Project Structure
```bash
/data                - Raw and processed data files
/docs                - Project documentation and reports
/notebooks           - Jupyter notebooks for analysis, exploration, and modeling
/src                 - Source code for model training, evaluation, and prediction
/environment.yml     - Conda environment configuration
/README.md           - Project overview and instructions
```

## Key Findings

- The project evaluates multiple machine learning models, with the **LightGBM (LGBM)** algorithm providing the best performance.
- The model achieved an **RMSE of 1037** using all features without scaling.
- Hyperparameter tuning further reduced the RMSE, demonstrating the importance of tuning for improving model accuracy.