# %%
# Importing Libraries
# Data manipulation
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")

# ML Models
import xgboost as xgb

# Metrics
from sklearn.metrics import mean_squared_error

# %%
# Importing data
df0 = pd.read_csv("./data/raw/AEP_hourly.csv.zip")

# Setting index
df0 = df0.set_index("Datetime")
df0.head()

# %%
# Checking index data type
df0.index.dtype

# %%
# Converting index to datetime
# ISO8601 for date in the format YYYY-MM-DD
df0.index = pd.to_datetime(df0.index, format="ISO8601")
df0.index.dtype

# %%
# Checking for the first and last data collected in the dataset
df0.index.min(), df0.index.max()

# %%
df0.columns = ["energy_mw"]
df0.head()

# %%
# Visualization of the entire energy consumption
df0.plot(
    style=".",
    figsize=(15,5),
    title="Energy Consumption in MW"
)

# %%
# Checking the distribution on a specific month
# 
(df0.loc[(df0.index > "01-01-2010") & (df0.index < "02-01-2010")]
 .plot(figsize=(15,5), 
       title="Monthly Energy Consumption"))

plt.show()

# %%
# Data seems to be disorganized. Sorting the Data.
df0 = df0.sort_index()
df0.head()

# %%
# Checking the distribution on a specific month
(df0.loc[(df0.index > "01-01-2010") & (df0.index < "01-12-2010")]
 .plot(figsize=(15,5), 
       title="Monthly Energy Consumption"))

plt.show()

# %%
# Splitting the dataframe
# DO NOT SPLIT RANDOMLY. This is a TIME SERIES, the split must be a temporal window in the dataframe.

# Train dataframe (up to the end of 2014)
df_train = df0.loc[df0.index < "2015-01-01"]

# Test dataframe (from 2015 onwards)
df_test = df0.loc[df0.index >= "2015-01-01"]

# %%
# Checking values
df_train.index.min(), df_train.index.max()

# %%
# Checking values
df_test.index.min(), df_test.index.max()

# %%
# Checking the splitting
fig, ax = plt.subplots(figsize=(15, 5))
df_train.plot(ax=ax, 
              label="Train", 
              title="Train/Test Split")

df_test.plot(ax=ax, 
             label="Test")

ax.axvline("2015-01-01", 
           color="black", 
           ls="--")

ax.legend(["Train", "Test"])
plt.show()

# %%
# Creating a function to add features
# Index MUST be a Datetime data type
def date_features(df):
    # Create a copy of the dataframe
    df = df.copy()

    # Creating Features
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["week_of_the_year"] = df.index.isocalendar().week
    df["day_of_year"] = df.index.dayofyear
    df["day_of_month"] = df.index.day
    df["day_of_week"] = df.index.dayofweek
    df["hour"] = df.index.hour

    return df

# %%
# Adding features
df0 = date_features(df0)
df0.head()

# %%
# Boxplot the relation on Energy Consumption x Hour
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(data=df0, 
            x="hour", 
            y="energy_mw", 
            palette="pastel", 
            hue="hour", 
            legend=False)
ax.set_title("Energy(MW) x Hour")
plt.show()

# %%
# Boxplot the relation on Energy Consumption x Month
fig, ax = plt.subplots(figsize=(15, 5))
sns.boxplot(data=df0,
            x= "month",
            y= "energy_mw", 
            palette="pastel", 
            hue="month", 
            legend=False)
ax.set_title("Energy(MW) x Month")
plt.show()

# %%
# Function to create time lag features
def time_lag(df):
    # Create a copy of the dataframe
    df = df.copy()

    # 1 hour before
    df["energy_mw_1h"] = df["energy_mw"].shift(1)
    
    # 1 day before
    df["energy_mw_1d"] = df["energy_mw"].shift(1*24)

    # 1 week before
    df["energy_mw_1w"] = df["energy_mw"].shift(1*24*7)

    return df

# %%
# Adding features
df0 = time_lag(df0)
df0.head(25)

# %%
# Model creation
# Adding features to Train and Test bases
df_train = date_features(df_train)
df_train = time_lag(df_train)

df_test = date_features(df_test)
df_test = time_lag(df_test)

# %%
# Setting variables
TARGET_VARIABLE = "energy_mw"
print(TARGET_VARIABLE)

y_train = df_train[TARGET_VARIABLE]
y_test = df_test[TARGET_VARIABLE]

PREDICTOR_VARIABLES = [cols for cols in df0.columns]
PREDICTOR_VARIABLES.remove(TARGET_VARIABLE)
print(PREDICTOR_VARIABLES)

X_train = df_train[PREDICTOR_VARIABLES]
X_test = df_test[PREDICTOR_VARIABLES]

# %%
# Instancing XGBoost
reg_xgb = xgb.XGBRegressor(base_score= 0.5,
                           booster= "gbtree",
                           n_estimators= 1000,
                           early_stopping_rounds= 50,
                           objective= "reg:linear",
                           max_depth=3,
                           learning_rate=0.01)

# %%
# Training model
reg_xgb.fit(X_train, 
            y_train, 
            eval_set= [(X_train, y_train), (X_test, y_test)],
            verbose=100)

# %%
# Calculating Feature Importance
feature_imp = pd.DataFrame(data= reg_xgb.feature_importances_,
                                  index= reg_xgb.feature_names_in_,
                                  columns= ["feature_importance"])

feature_imp.sort_values("feature_importance").plot(
    kind="barh",
    title="Feature Importance")

plt.show()

# %%
# Adding predictions to the test dataframe
df_test["energy_mw_predictions"] = reg_xgb.predict(X_test)
df_test.head()

# %%
# Adding the test predictions to original dataframe to visualize the difference
df0 = df0.merge(df_test[["energy_mw_predictions"]],
                how="left",
                left_index=True,
                right_index=True)

df0.tail()

# %%
# Plot to verify the predictions
fig, ax = plt.subplots(figsize=(15, 5))
df0["energy_mw"].plot(ax=ax, 
                      label="Real", 
                      title="Real Values x Predict Values")

df0["energy_mw_predictions"].plot(ax=ax, 
                                  label="Predict",
                                  alpha=0.5)

ax.axvline("2015-01-01", 
           color="black", 
           ls="--")

ax.legend(["Real", "Predict"])
plt.show()
# %%
# Plotting a week comparison
fig, ax = plt.subplots(figsize=(15,5))
(df0.loc[(df0.index > "2017-10-20")
          & 
          (df0.index < "2017-10-27")]["energy_mw"]
          .plot(ax=ax,
                label="Real",
                title="Real Values x Predict Values (Week)"))

(df0.loc[(df0.index > "2017-10-20")
         &
         (df0.index < "2017-10-27")]["energy_mw_predictions"]
         .plot(ax=ax,
               label="Predict",
               style="."))
plt.legend(["Real", "Predict"])
plt.show()

# %%
# Calculate the error (RMSE)
score = np.sqrt(mean_squared_error(df_test[TARGET_VARIABLE], df_test["energy_mw_predictions"]))

print(f"RMSE Score in the test Dataframe: {score:0.2f}")

# %%
# Calculating the absolute error
df_test["error"] = np.abs(df_test[TARGET_VARIABLE] - df_test["energy_mw_predictions"])

# Grouping by day
df_test.groupby(["year", "month", "day_of_month"])["error"].mean().sort_values(ascending=False).head(10)
# %%
