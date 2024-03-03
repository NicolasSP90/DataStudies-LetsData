# %%
# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle as pk

# %%
# Importing dataframe
df0 = pd.read_csv("./data/train.csv")

df0.head()

# %%
# Separating predictor and target features
X = df0.drop(["Survived", "PassengerId"], axis="columns")
display(X.head())

y = df0["Survived"]
display(y.head())

# %%
# Splitting BEFORE any transformation helps prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%
# Checking data types
X_train.info()

# %%
# Separating categorical and numerical features
cols_num = [col for col in X_train.columns if X_train[col].dtype in ["float64", "int64"]]
display(cols_num)

cols_categ = [col for col in X_train.columns if X_train[col].dtype == "object"]
display(cols_categ)

# %%
# Creating a transformation pipeline for numeric features
# First it wil perform data imputation (SimpleInputer), than scaling (StandardScaler)
transf_numeric = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
    ])

display(transf_numeric)

# %%
# Creating a transformation pipeline for categorical features
# First it will perform data imputation (SimpleInputer), than encoding the columns(OneHotEncoder)
transf_categorical = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("OHE", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

display(transf_categorical)

# %%
# Instancing the transformations
pre_processor = ColumnTransformer(
    transformers=[
        ("numeric", transf_numeric, cols_num),
        ("categorical", transf_categorical, cols_categ)
    ]
)

display(pre_processor)

# %%
# Complete Pipeline with transformation (pre-processing) and model creation
pipeline_logreg = Pipeline(steps=[
    ("preprocessor", pre_processor),
    ("model", LogisticRegression(max_iter=300, random_state=42))
])

display(pipeline_logreg)

# %%
# Its easier to create multiple models, by calling the pipeline
pipeline_histgrad = Pipeline(steps=[
    ("preprocessor", pre_processor),
    ("model", HistGradientBoostingClassifier(random_state=42))
])

display(pipeline_histgrad)

# %%
# Cross validation to check between models
score_logreg = cross_val_score(pipeline_logreg, X_train, y_train, cv=5)
display(score_logreg.mean())

score_histgrad = cross_val_score(pipeline_histgrad, X_train, y_train)
display(score_histgrad.mean())

# %%
# Determine the best model
best_pipeline = pipeline_logreg if score_logreg.mean() > score_histgrad.mean() else pipeline_histgrad

display(best_pipeline)
# %%
# Fitting train bases in the pipeline
best_pipeline.fit(X_train, y_train)

# %%
# Predictions using the pipeline
predictions = best_pipeline.predict(X_test)

display(predictions)
# %%
# Checking accuracy
accuracy_score(y_test, predictions)
# %%
# Using pickle to storne the pipeline

with open('./models/histgrad-titanic.pickle',"wb") as model_file:
    pk.dump(best_pipeline, model_file)