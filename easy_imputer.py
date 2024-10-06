import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class EasyImputer(BaseEstimator, TransformerMixin):
    """
    This class manually fills missing values without using sklearn's SimpleImputer.

    Parameters:
    ----------------------------------

    numeric_only: bool, default=True
    When set to False, it fills in missing values in both numeric variables and
    categorical variables. If True, only numeric variables will be imputed.

    missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
    The placeholder for the missing values. All occurrences of `missing_values` will be imputed. 

    strategy: str, default='mean' for numeric columns
    The imputation strategy:
        - "mean": Replace missing values using the mean along each column (numeric data only).
        - "median": Replace missing values using the median along each column (numeric data only).
        - "most_frequent": Replace missing values using the most frequent value along each column (can be used for numeric and categorical data).
        - "constant": Replace missing values with `fill_value` (can be used for numeric and categorical data).

    fill_value : str or numerical value, default=None
    When strategy == "constant", `fill_value` is used to replace all occurrences of missing_values. 
    If `None`, `fill_value` will be 0 for numeric data and "missing_value" for categorical data.

    copy : bool, default=True
    If True, a copy of X will be created. If False, imputation will be done in-place when possible.
    """

    def __init__(self, numeric_only=True, missing_values=np.nan, strategy="mean", fill_value=None, copy=True):
        self.numeric_only = numeric_only
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy

        # Validate strategy
        valid_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {self.strategy}. Choose from {valid_strategies}")

    def fit(self, X, y=None):
        """
        Fit the imputer on the dataset. Calculate the necessary statistics for imputing missing values.
        
        Parameters:
        ----------------------------------
        X: pandas DataFrame
        The data to be used to calculate imputation values.

        y: Ignored. Present for compatibility with sklearn pipeline.
        """

        # Detect numeric columns
        self.num_cols = X.select_dtypes(exclude=["object", "category"]).columns

        if not self.numeric_only:
            # Detect categorical columns
            self.cat_cols = X.select_dtypes(include=["object", "category"]).columns
        else:
            self.cat_cols = None

        # Dictionary to store the statistics for imputation
        self.statistics_ = {}

        # Calculate statistics for numeric columns based on the selected strategy
        if self.strategy == "mean":
            self.statistics_["num"] = X[self.num_cols].mean()
        elif self.strategy == "median":
            self.statistics_["num"] = X[self.num_cols].median()
        elif self.strategy == "most_frequent":
            self.statistics_["num"] = X[self.num_cols].mode().iloc[0]
        elif self.strategy == "constant":
            self.statistics_["num"] = pd.Series(self.fill_value, index=self.num_cols)

        # If categorical columns exist, calculate the most frequent value for categorical data
        if self.cat_cols is not None:
            self.statistics_["cat"] = X[self.cat_cols].mode().iloc[0]

        return self

    def transform(self, X):
        """
        Impute missing values in the dataset.

        Parameters:
        ----------------------------------
        X: pandas DataFrame
        The input data to be transformed.

        Returns:
        ----------------------------------
        pandas DataFrame
        A DataFrame where the missing values have been imputed based on the fit statistics.
        """

        # Create a copy of the data if required
        X_copy = X.copy() if self.copy else X

        # Fill missing values in numeric columns
        X_copy[self.num_cols] = X_copy[self.num_cols].fillna(self.statistics_["num"])

        # If there are categorical columns, fill missing values in categorical columns
        if self.cat_cols is not None:
            X_copy[self.cat_cols] = X_copy[self.cat_cols].fillna(self.statistics_["cat"])

        # Check for any columns that remain with missing values and issue a warning
        if X_copy.isnull().any().any():
            print("Warning: Some columns still have missing values after imputation.")

        return X_copy

    def fit_transform(self, X, y=None):
        """
        Fit the imputer and then transform the dataset in a single step.

        Parameters:
        ----------------------------------
        X: pandas DataFrame
        The input data to fit and transform.

        Returns:
        ----------------------------------
        pandas DataFrame
        A DataFrame where the missing values have been imputed based on the fit statistics.
        """
        return self.fit(X).transform(X)