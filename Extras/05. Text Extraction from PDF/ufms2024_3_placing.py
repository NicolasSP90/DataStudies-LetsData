#%%
import pandas as pd
# %%
df_courses = pd.read_csv("./data/df_courses.csv")
df_courses.head()
# %%
df_courses = df_courses.drop(["Unnamed: 0"], axis=1)
df_courses.head()
#%%
df_courses.shape
df_courses.info()
# %%
df_grades = pd.read_csv("./data/df_grades.csv")
df_grades.head()
# %%
df_grades = df_grades.drop(["Unnamed: 0"], axis=1)
df_grades.head()
#%%
df_grades.shape
df_grades.info()
# %%
df_general = pd.merge(df_grades, df_courses, how="inner", on=["id"])
df_general.shape
# %%
df_general.head()
# %%
df_general.isna().value_counts()
# %%
# %%
myid = 963760
totalcandidates = df_general.shape[0]
df_general[df_general["id"] == myid]
# %%
myvl = df_general.loc[df_general[df_general["id"] == myid].index[0], "VL"]
myvl
# %%
myvh = df_general.loc[df_general[df_general["id"] == myid].index[0], "VH"]
myvh
# %%
myvn = df_general.loc[df_general[df_general["id"] == myid].index[0], "VN"]
myvn
# %%
myvm = df_general.loc[df_general[df_general["id"] == myid].index[0], "VM"]
myvm
# %%
myred = df_general.loc[df_general[df_general["id"] == myid].index[0], "RED"]
myred
#%%
df_AC_DS = df_general[(df_general["type"] == "AC") & (df_general["course"] == "CIÊNCIADOSDADOS")]

df_AC_DS.head()
#%%
df_AC_DS.shape
#%%
# VL
(df_AC_DS["VL"] >= myvl).value_counts()[True]
# %%
(df_AC_DS["VH"] >= myvh).value_counts()[True]
# %%
(df_AC_DS["VN"] >= myvn).value_counts()[True]
# %%
(df_AC_DS["VM"] >= myvm).value_counts()[True]
# %%
(df_AC_DS["RED"] >= myred).value_counts()[True]
# %%
