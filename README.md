# EasyImputer
Fills dataframes with missing data according to specific strategies
## Introduction
Handling missing data is a common preprocessing step in machine learning projects. Sklearn's SimpleImputer is a popular tool for filling missing values, but there may be scenarios where you need more customization. This project introduces SımpleImputer, a flexible Python class that allows you to fill missing values in both numeric and categorical features using a variety of imputation strategies.

In this guide, I will explain how the EasyImputer works, the various parameters it supports, and how you can easily integrate it into your machine learning pipeline. We will also explore a step-by-step example to demonstrate its functionality.

## Key Features
* Supports both numeric and categorical data imputation.
* Various imputation strategies such as mean, median, most frequent, and constant.
* Handles missing values in both training and test datasets.
* Fully compatible with sklearn pipelines (fit, transform, and fit_transform methods are implemented).
* Optionally copies the original data, allowing in-place transformations if desired.

## Class Overview: EasyImputer
The EasyImputer class is designed to manually fill missing values based on the strategy you define. It works similarly to SimpleImputer but offers more flexibility and control.

## Parameters:
**numeric_only (bool, default=True):** If set to False, the imputer will handle both numeric and categorical features. Otherwise, only numeric features are processed.

**missing_values (int, float, str, np.nan, None, or pd.NA, default=np.nan):** The placeholder for missing values in the dataset. You can define any type of missing value (e.g., np.nan, None, or pd.NA).

**strategy (str, default='mean'):** The imputation strategy. Available options:

"mean": Replace missing values using the mean of each numeric column.

"median": Replace missing values using the median of each numeric column.

"most_frequent": Replace missing values using the most frequent value. This can be applied to both numeric and categorical data.

"constant": Replace missing values with a constant value defined by fill_value.

fill_value (str or numerical value, default=None): When strategy="constant", this value will be used to replace missing values. If not set, it defaults to 0 for numeric data and "missing_value" for strings or object data types.

**copy (bool, default=True):** If True, a copy of the dataset will be created for transformations. If False, the imputation will modify the dataset in place.

## Methods
**fit(X, y=None):** Fits the imputer to the dataset, calculating the necessary statistics (mean, median, mode, or constant) based on the selected strategy.

**transform(X):** Applies the imputation to the dataset, replacing missing values with the calculated statistics.

**fit_transform(X, y=None):** Combines fit and transform into a single step, fitting the imputer to the data and then transforming it.


# Detailed Explanation of Functionality
## 1. Initialization
Upon creating a SımpleImputer object, you define the imputation strategy, the type of missing values, and whether the imputer should handle only numeric data or both numeric and categorical data.

```python
from easy_imputer import EasyImputer
imputer = EasyImputer(numeric_only=False, strategy="most_frequent")
```

This creates an imputer that will replace missing values in both numeric and categorical features with the most frequent value.

## 2. Fitting the Data
The fit method is used to calculate the statistics necessary for imputation. For numeric data, depending on the strategy, it calculates either the mean, median, or most frequent value. For categorical data, it calculates the most frequent value.
```python
imputer.fit(X)
```
Here, X is the DataFrame on which the imputer will be trained.

## 3. Transforming the Data
Once the imputer is fitted, you can use the transform method to apply the imputation to a new dataset.
```python
X_imputed = imputer.transform(X)
```
This will replace all missing values in X with the imputed values based on the strategy you defined.

## 4. Handling Both Numeric and Categorical Data
If numeric_only=False is specified, the imputer handles both numeric and categorical data. For numeric columns, the strategy you choose (mean, median, etc.) will be applied. For categorical columns, the most frequent value will be used.

# Example Usage
Let's walk through an example to show how you can use the EasyImputer in practice.
```python
import pandas as pd
from easy_imputer import EasyImputer

# Create a sample dataset with missing values
data = {
    'Age': [25, np.nan, 35, np.nan],
    'Gender': ['Male', 'Female', np.nan, 'Female'],
    'Income': [50000, np.nan, 60000, 70000]
}

df = pd.DataFrame(data)

# Initialize the imputer for both numeric and categorical data
imputer = EasyImputer(numeric_only=False, strategy="most_frequent")

# Fit the imputer and transform the dataset
imputed_df = imputer.fit_transform(df)

print(imputed_df)

"""
    Age  Gender  Income
0  25.0    Male  50000.0
1  25.0  Female  60000.0
2  35.0    Male  60000.0
3  25.0  Female  70000.0

"""
```
In this example:

* Missing values in the Age column were replaced with the most frequent value 25.0.
* Missing values in the Gender column were replaced with the most frequent value Male.
* Missing values in the Income column were replaced with the most frequent value 60000.0.


# Why Use EasyImputer?
While SimpleImputer from sklearn is a great tool, there may be cases where you need more flexibility in your data preprocessing pipelines. EasyImputer provides:

**Customization:** You can easily add or modify the imputation logic without relying on external libraries.

**Pipeline Compatibility:** It integrates seamlessly with sklearn pipelines, making it ideal for machine learning workflows.

**Handling Mixed Data Types:** It supports datasets with both numeric and categorical features, making it versatile for different scenarios.

# Conclusion
The EasyImputer is a highly flexible and customizable tool for handling missing data. Whether you are working with numeric or categorical data, or need more control over the imputation strategy, this class can be easily integrated into your projects.

Feel free to explore the code and adapt it to your needs. Contributions and suggestions are welcome!

I hope this detailed explanation helps you better understand the structure and usage of the EasyImputer. Whether you're building a machine learning pipeline or working on data preprocessing tasks, this tool can save you time and provide greater control over missing data handling.

