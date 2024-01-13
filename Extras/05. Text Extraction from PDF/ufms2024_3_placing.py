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
df_my = df_general[df_general["id"] == myid]
df_my = df_my.set_index("id")
df_my.head()
#%%
totalcandidates = df_general[df_general["course"] == df_my.loc[myid, "course"]].shape[0]
totalcandidates
#%%
(df_general["VL"] >= df_my.loc[myid,"VL"]).value_counts()[True]
# %%
(df_general["VH"] >= df_my.loc[myid,"VH"]).value_counts()[True]
# %%
(df_general["VN"] >= df_my.loc[myid,"VN"]).value_counts()[True]
# %%
(df_general["VM"] >= df_my.loc[myid,"VM"]).value_counts()[True]
# %%
(df_general["RED"] >= df_my.loc[myid,"RED"]).value_counts()[True]
#%%
#%%
df_DS = df_general[(df_general["course"] == "CIÊNCIADOSDADOS")]
df_DS.head()
#%%
df_DS.shape
#%%
(df_DS["VL"] >= df_my.loc[myid,"VL"]).value_counts()[True]
# %%
(df_DS["VH"] >= df_my.loc[myid,"VH"]).value_counts()[True]
# %%
(df_DS["VN"] >= df_my.loc[myid,"VN"]).value_counts()[True]
# %%
(df_DS["VM"] >= df_my.loc[myid,"VM"]).value_counts()[True]
# %%
(df_DS["RED"] >= df_my.loc[myid,"RED"]).value_counts()[True]
#%%
df_AC_DS = df_general[(df_general["type"] == "AC") & (df_general["course"] == "CIÊNCIADOSDADOS")]
df_AC_DS.head()
#%%
df_AC_DS.shape
#%%
# VL
(df_AC_DS["VL"] >= df_my.loc[myid,"VL"]).value_counts()[True]
# %%
(df_AC_DS["VH"] >= df_my.loc[myid,"VH"]).value_counts()[True]
# %%
(df_AC_DS["VN"] >= df_my.loc[myid,"VH"]).value_counts()[True]
# %%
(df_AC_DS["VM"] >= df_my.loc[myid,"VM"]).value_counts()[True]
# %%
(df_AC_DS["RED"] >= df_my.loc[myid,"RED"]).value_counts()[True]
# %%
