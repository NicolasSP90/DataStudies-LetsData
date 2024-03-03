# Pipelines using Scikit-Learn
The main goals of this study are:
<li>Understand Pipelines (Scikit-Learn) in practical examples
<li>Create a structure of training and validation of ML models with pipelines
<li>Creating the habit of always create preprocessing pipelines

Some advantages of using pipelines:
<li>Organization of important steps in during the creation of ML models
<li>Guarantee that the preprocessing steps for creating the ML model will be applied in the train and test steps
<li>Less coding (not need to repeat the same code over and over)
<li>Avoid data leakage
<li>Easier to update / swap ML models
<li>Saving and Loading as an object
<li>Allows several preprocessing steps (transform)
<li>Usually finishes with a predictive algorithm (estimator)
<li>Allows the selection of features for each preprocessing step
<li>Creates an path that the raw data will follow
<li>Again, avoid data leakage

The dataframe used will be from Kaggle's Titanic competition, for an easy understanding.

Side note: The focus is to understand the Pipelines from Scikit-Learn, not to create a fully functional model with every EDA step needed.