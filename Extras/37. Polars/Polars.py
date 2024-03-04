# %%
# Importing Libraries
import polars as pl
import pandas as pd
import numpy as np
import time

# %%
# Creating a generic dataframe
data = {
    "A": np.random.rand(100000),
    "B": np.random.rand(100000),
    "C": np.random.randint(0,100,100000)
}

# %%
# Storing in a Pandas Dataframe
df_pandas = pd.DataFrame(data=data)

display(df_pandas.shape)

display(df_pandas.head())

display(df_pandas.info())

# %%
# Storing in a Polars Dataframe
df_polars = pl.DataFrame(data=data)

display(df_polars.shape)

display(df_polars.head())

display(df_polars.estimated_size(unit="mb"))

# %%
# Filter with Pandas
pandas_filter = df_pandas.loc[df_pandas["A"] < 0.5]

display(pandas_filter)

# %%
# Filter with Polars
polars_filter = df_polars.filter(df_polars["A"] < 0.5)

display(polars_filter)

# %%
# Aggregation and Groupping with Pandas
pandas_agg = df_pandas.groupby("C").agg({"A": "mean", "B": "sum"})

pandas_agg

# %%
# Aggregation and Groupping with Polars
polars_agg = df_polars.group_by("C").agg(
    [
        pl.col("A").mean(), 
        pl.col("B").sum()
        ])

display(polars_agg)

# %%
# Merging Dataframes - Dataframe creation
data_2 = {
    "C": np.random.randint(0,100,100000),
    "D": np.random.rand(100000)
}

# %%
# Pandas dataframe to be merged
df_pandas_2 = pd.DataFrame(data=data_2)

display(df_pandas_2.shape)

display(df_pandas_2.head())

display(df_pandas_2.info())

# %%
# Polars dataframe to be merged
df_polars_2 = pl.DataFrame(data=data_2)

display(df_polars_2.shape)

display(df_polars_2.head())

display(df_polars_2.estimated_size(unit="mb"))

# %%
# Merging Dataframes - Pandas
df_pandas_join = df_pandas.merge(df_pandas_2, on="C")

display(df_pandas_join.shape)

display(df_pandas_join.head())

display(df_pandas_join.info())

# %%
# Merging Dataframes - Pandas
df_polars_join = df_polars.join(df_polars_2, on="C")

display(df_polars_join.shape)

display(df_polars_join.head())

display(df_polars_join.estimated_size(unit="mb"))

# %%
# Reading CSV
df_pandas_csv = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/raw/taxis.csv")

df_polars_csv = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/raw/taxis.csv")

# %%
# Reading Parquet - Pandas
df_pandas_parquet = pd.read_parquet("./data/enem_2019_10.parquet")

display(df_pandas_parquet.shape)

display(df_pandas_parquet.head())

display(df_pandas_parquet.info())

# %%
df_polars_parquet = pl.read_parquet("./data/enem_2019_10.parquet")

display(df_polars_parquet.shape)

display(df_polars_parquet.head())

display(df_polars_parquet.estimated_size(unit="mb"))

# %%
# Speed - Agregation
start_time = time.time()
pandas_agg = df_pandas.groupby("C").agg({"A": "mean", "B": "sum"})
pandas_time = time.time() - start_time

start_time = time.time()
polars_agg = df_polars.group_by("C").agg([
    pl.col("A").mean(), 
    pl.col("B").sum()
    ])
polars_time = time.time() - start_time

print(f"Pandas time: {pandas_time}")
print(f"Polars time: {polars_time}")

# %%
# Speed - Parquet
start_time = time.time()
df_pandas_parquet = pd.read_parquet("./data/enem_2019_10.parquet")
pandas_time = time.time() - start_time

start_time = time.time()
df_polars_parquet = pl.read_parquet("./data/enem_2019_10.parquet")
polars_time = time.time() - start_time

print(f"Pandas time: {pandas_time}")
print(f"Polars time: {polars_time}")

# %%
# Speed - Agregation (On a more "real" base)
start_time = time.time()
df_pandas_parquet.groupby("TP_SEXO").agg({"NU_NOTA_REDACAO": "mean", "NU_NOTA_COMP1": "sum"})
pandas_time = time.time() - start_time

start_time = time.time()
df_polars_parquet.groupby("TP_SEXO").agg([ pl.col("NU_NOTA_REDACAO").mean(), pl.col("NU_NOTA_COMP1").sum()])
polars_time = time.time() - start_time

print(f"Pandas time: {pandas_time}")
print(f"Polars time: {polars_time}")
# %%
# Dataframe Taxis (with extra features)
df_taxis = pl.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/raw/taxis.csv")

df_taxis.head()

# %%
# Filter
df_taxis.filter(pl.col("passenger_count") > 5)

# %%
# Sorting column
df_taxis.sort("trip_distance")

# %%
# Trip Distance
df_taxis.sort("trip_distance", descending=T)

# %%
# Select
df_taxis.select(pl.col('trip_distance', 'passenger_count'))

# %%
df_taxis.select(pl.exclude('trip_distance', 'passenger_count'))
# %%
