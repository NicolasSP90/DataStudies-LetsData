# %%
# Import Libraries
import pandas as pd
from pycaret.datasets import get_data
from pycaret.classification import *
from sklearn.model_selection import train_test_split
# %%
# Importing preloaded dataframe
df_diabetes = get_data("diabetes")
df_diabetes
# %%
# Chcking for basic statistics
df_diabetes.describe()
# %%
# Checking people with diabetes (in %)
100 * len(df_diabetes.loc[df_diabetes["Class variable"] == 1]) / len(df_diabetes)
# %%
# Base metric for validation of accuracy
100 - (100 * len(df_diabetes.loc[df_diabetes["Class variable"] == 1]) / len(df_diabetes))
# %%
# Splitting bases
X_train, X_test = train_test_split(df_diabetes, test_size=0.2, random_state=42)
print(f"Train data: {X_train.shape}")
print(f"Train data: {X_test.shape}")
# %%
# First setup
experiment_01 = setup(data = X_train, target = "Class variable", session_id=123, )
# %%
