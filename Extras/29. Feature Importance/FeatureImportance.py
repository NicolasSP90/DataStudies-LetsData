# %%
# Importing libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# %%
# Instancing iris data
iris = load_iris()

# Storing iris data in a pandas dataframe
df_iris = pd.DataFrame(data = iris.data, 
                       columns=iris.feature_names)

df_iris.head()

# %%
# Adding a new column to the dataframe
df_iris["species"] = iris.target

df_iris.head()

# %%
# Counting values from species column
df_iris["species"].value_counts(dropna=False)

# %%
# Separating bases
X, y = iris.data, iris.target

# %%
# Logistic Regression model to check feature importance
clf_logreg = LogisticRegression(random_state=42).fit(X, y)

# Using the coeficients to check feature importance
importance_clf_logreg = clf_logreg.coef_[0]

for i, v in enumerate(importance_clf_logreg):
    print("Feature %s, Score %.5f" % (iris.feature_names[i], v))

# %%
# Random Forest model to check feature importance
clf_ranfor = RandomForestClassifier(random_state=42).fit(X,y)

# Using the feature importances feature
importance_clf_ranfor = clf_ranfor.feature_importances_

for i, v in enumerate(importance_clf_ranfor):
    print("Feature %s, Score %.5f" % (iris.feature_names[i], v))

# %%
# Random Forest model to check feature importance - With Permutation Importance
results = permutation_importance(clf_ranfor, X, y, scoring="accuracy")

# Using the importances mean
importance_clf_ranfor_perm_imp = results.importances_mean

for i, v in enumerate(importance_clf_ranfor_perm_imp):
    print("Feature %s, Score %.5f" % (iris.feature_names[i], v))

# %%
# LASSO Regression model to check feature selection
clf_LASSO = Lasso(alpha=0.1).fit(X,y)

# Using the coeficients to check feature importance
importance_clf_LASSO = clf_LASSO.coef_

for i, v in enumerate(importance_clf_LASSO):
    print("Feature %s, Score %.5f" % (iris.feature_names[i], v))

# %%
# Using Support Vector Regression (SVR) algorith with Recursive Feature Elimination (RFE) to generate a model to check for feature selection
# SVR is used to find the hyperplanes in n-dimensional features
# RFE recursively fitts a model and remove the least important features

estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=2, step=1)
selector = selector.fit(X,y)

for i, feature in enumerate(iris.feature_names):
    if selector.support_[i]:
        print("Feature: ", feature)

# %%
# Creating barplot to see feature importance
plt.figure(figsize=(10, 5))
plt.bar(range(X.shape[1]), importance_clf_ranfor)
plt.xticks(range(X.shape[1]), iris.feature_names, rotation=20)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.show()
# %%
