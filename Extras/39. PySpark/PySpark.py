# %%
# Importing Libraries
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, max, min, sum, count, rank, lead, lag

# %%
# Setting PySpark Context
sc = SparkContext('local')
spark = SparkSession(sc)

# %%
# Creating a Spark Session to initialize PySpark
session = SparkSession.builder.appName("LeaningSpark").getOrCreate()

# %%
# Checking session data type
type(session)

# %%
# Loading data
file = r".\data\raw\employees.csv"

df = spark.read.csv(file, header=True, inferSchema=True, sep="\t")

# %%
# Checkign the Dataframe - Similar to pandas' .info()
df.printSchema()

# %%
# Checking df rows
df.show(10)

# %%
# Grouping and Aggregation methods
df.groupBy("Sex").agg(
    avg("Salary").alias("Average Salary"),
    sum("Salary").alias("Salary Sum"),
    count("Salary").alias("Employees Numbers")
).show()

# %%
# Filtering - Similar with Polars and Pandas (loc, iloc)
df.filter((df["Sex"] == "Female") & (df["Salary"] > 1500)).show()

# %%
