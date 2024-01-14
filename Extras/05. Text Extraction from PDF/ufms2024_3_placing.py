#%%
# Importing Library
import pandas as pd
#%%
# Creating Dataframe of courses
df_courses = pd.read_csv("./data/df_courses.csv")
df_courses.head()
#%%
# Removing unwanted columns
df_courses = df_courses.drop(["Unnamed: 0"], axis=1)
df_courses.head()
#%%
# Checking Dataframe size
df_courses.shape
df_courses.info()
#%%
# Creating Dataframe of grades
df_grades = pd.read_csv("./data/df_grades.csv")
df_grades.head()
#%%
# Removing unwanted columns
df_grades = df_grades.drop(["Unnamed: 0"], axis=1)
df_grades.head()
#%%
# Checking Dataframe size
df_grades.shape
df_grades.info()
#%%
# Merging Dataframes
df_general = pd.merge(df_grades, df_courses, how="inner", on=["id"])
df_general.shape
#%%
# Head Visualization
df_general.head()
#%%
# Checking for empty values
df_general.isna().value_counts()
#%%
# Creating a Dataframe for my situation
myid = 963760
df_my = df_general[df_general["id"] == myid]
df_my = df_my.set_index("id")
df_my.head()
#%%
# Filtering all candidates that applied for the same course as me
df_general[df_general["course"] == df_my.loc[myid, "course"]].shape[0]
#%%
# My position considering my VL grade with all candidates
(df_general["VL"] >= df_my.loc[myid,"VL"]).value_counts()[True]
#%%
# My position considering my VH grade with all candidates
(df_general["VH"] >= df_my.loc[myid,"VH"]).value_counts()[True]
#%%
# My position considering my VN grade with all candidates
(df_general["VN"] >= df_my.loc[myid,"VN"]).value_counts()[True]
#%%
# My position considering my VM grade with all candidates
(df_general["VM"] >= df_my.loc[myid,"VM"]).value_counts()[True]
#%%
# My position considering my RED grade with all candidates
(df_general["RED"] >= df_my.loc[myid,"RED"]).value_counts()[True]
#%%
# Creating Datafame of DS candidates
df_DS = df_general[(df_general["course"] == "CIÊNCIADOSDADOS")]
df_DS.head()
#%%
# Checking DS Dataframe size
df_DS.shape
#%%
# My position considering my VL grade with all DS candidates
(df_DS["VL"] >= df_my.loc[myid,"VL"]).value_counts()[True]
#%%
# My position considering my VH grade with all DS candidates
(df_DS["VH"] >= df_my.loc[myid,"VH"]).value_counts()[True]
#%%
# My position considering my VN grade with all DS candidates
(df_DS["VN"] >= df_my.loc[myid,"VN"]).value_counts()[True]
#%%
# My position considering my VM grade with all DS candidates
(df_DS["VM"] >= df_my.loc[myid,"VM"]).value_counts()[True]
#%%
# My position considering my RED grade with all DS candidates
(df_DS["RED"] >= df_my.loc[myid,"RED"]).value_counts()[True]
#%%
# Creating Datafame of DS candidates without reserved positions
df_AC_DS = df_general[(df_general["type"] == "AC") & (df_general["course"] == "CIÊNCIADOSDADOS")]
df_AC_DS.head()
#%%
# Checking Dataframe size
df_AC_DS.shape
#%%
# My position considering my VL grade with all DS candidates without reserved positions
(df_AC_DS["VL"] >= df_my.loc[myid,"VL"]).value_counts()[True]
#%%
# My position considering my VH grade with all DS candidates without reserved positions
(df_AC_DS["VH"] >= df_my.loc[myid,"VH"]).value_counts()[True]
#%%
# My position considering my VN grade with all DS candidates without reserved positions
(df_AC_DS["VN"] >= df_my.loc[myid,"VN"]).value_counts()[True]
# %%
# My position considering my VM grade with all DS candidates without reserved positions
(df_AC_DS["VM"] >= df_my.loc[myid,"VM"]).value_counts()[True]
# %%
# My position considering my RED grade with all DS candidates without reserved positions
(df_AC_DS["RED"] >= df_my.loc[myid,"RED"]).value_counts()[True]
# %%
