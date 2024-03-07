# %%
# importing libraries
import duckdb
import pandas as pd
import os
import time

# %%
# Elapsed Time functions
def st_time():
    global str_time 
    str_time = time.time()

def lt_time():
    ltr_time = time.time()
    print(F"Elapsed time: {ltr_time - str_time}s")

# %%
# Create DuckDB connector
# Creates the database in memory
con = duckdb.connect()

# con = duckdb.connect("./data/raw/"DBNAME.duckdb"")

# %%
# Creating a table
# table "Workers", with 3 columns
con.execute("""
            CREATE TABLE workers (
            id INT,
            name VARCHAR,
            salary DECIMAL)
            """)

# %%
# Inserting data in the table
con.execute("""
            INSERT INTO workers (id, name, salary) VALUES
            (1, 'John', 3000),
            (2, 'Mary', 4000),
            (3, 'Anna', 3500)
            """)

# NOTE: MUST USE simple apostrophe ' instead of double "

# %%
# Basic query
result = con.execute("""
                     SELECT * 
                     FROM workers
                     """).fetchall()

# fetchall() returns a list of tuples
# fetchone() returns one tuple

result

# %%
# UPDATING database
con.execute("""
            UPDATE workers
            SET salary = 3800
            WHERE name = 'Anna'
            """)

result = con.execute("""
                     SELECT * 
                     FROM workers
                     """).fetchall()

result

# %%
# Deleting from database
con.execute("""
            DELETE FROM workers
            WHERE name = 'John'
            """)

result = con.execute("""
                     SELECT *
                     FROM workers
                     """).fetchall()

result

# %%
# Aggregation functions (mean of workers salary)
# Differences in the fetch() function returns.
mean_salary = con.execute("""
                          SELECT AVG(salary)
                          FROM workers""").fetchall()
display(mean_salary)

mean_salary = con.execute("""
                          SELECT AVG(salary)
                          FROM workers""").fetchone()
display(mean_salary)
display(mean_salary[0])

# %%
# Creating a second table called "departments"
con.execute("""
            CREATE TABLE departments(
            id INT,
            department_name VARCHAR
            )
            """)

con.execute("""
            INSERT INTO departments (id, department_name) VALUES
            (1, 'IT'),
            (2, 'Human Resources')
            """)

# %%
# Visualize the new table
result = con.execute("""
                     SELECT *
                     FROM departments
                     """).fetchall()

result

# %%
# Updating workers table to add department_id
con.execute("""
            ALTER TABLE workers 
            ADD COLUMN id_department INT
            """)

# Update values in the id_deartment column
con.execute("""
            UPDATE workers
            SET id_department = 1
            WHERE name = 'Mary'
            """)

con.execute("""
            UPDATE workers
            SET id_department = 2
            WHERE name = 'Anna'
            """)

result = con.execute("""
                     SELECT *
                     FROM workers
                     """).fetchall()

result

# %%
join_ = con.execute("""
                    SELECT w.name, d.department_name
                    FROM workers AS w
                    JOIN departments AS d ON w.id_department = d.id
                    """).fetchall()

join_

# %%
# USING PANDAS
all_csv = pd.concat([pd.read_csv(f"data/raw/{files_csv}") for files_csv in os.listdir("data/raw") if files_csv.endswith(".csv")])

all_csv.reset_index(drop=True)

all_csv

# %% 
# USING DUCKDB
con.execute("""
            SELECT *
            FROM 'data/raw/*.csv'
            """).fetchall()

# %%
# Query with DUCKDB and Storing in a Pandas dataframe
df_csv = con.execute("""
                     SELECT *
                     FROM 'data/raw/*.csv'
                     """).df()

df_csv

# %%
# Checking object type
type(df_csv)

# %%
# link of parquet file
link_parquet = 'https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-03.parquet'

# %%
# Time to download and read parquet file - PANDAS
st_time()

pd.read_parquet(link_parquet)

lt_time()

# %%
# Time to download and read parquet file - DuckDB
st_time()

duckdb.sql("""
           SELECT COUNT(*)
           FROM read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-03.parquet')
           """)

lt_time()

# %%
# Time show the first 10 rows - PANDAS
st_time()

pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-03.parquet').head(10)

lt_time()

# %%
# Time show the first 10 rows - DuckDB
st_time()

display(duckdb.sql("""
           SELECT *
           FROM read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-03.parquet')
           LIMIT 10
           """).df())

lt_time()

# %%
# Aggregate function time - PANDAS
st_time()

df = pd.read_parquet(link_parquet)

df.sort_values(by="hvfhs_license_num", ascending=False)

lt_time()

# %%
# Aggregate function time - DuckDB
st_time()

duckdb.sql("""
           SELECT dispatching_base_num, COUNT(*)
           FROM read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-03.parquet')
           GROUP BY dispatching_base_num
           ORDER BY 2 DESC""").df()

lt_time()

# %%
# The callings of FROM and SELECT do not need to be in the right order
con.execute("""
    FROM 'data/raw/*.csv'
    SELECT *
""").df()

# %%
# If the SELECT command is SELECT * only, it doenst need to called
con.execute("""
            FROM 'data/raw/*.csv'
            """).df()

# %%
# Regex can be used - Case 1
con.execute("""
            SELECT *
            FROM 'data/raw/*.csv'
            WHERE Nome like '%Smirnov'
            """).df()

# %%
# Regex can be used - Case 2 (.execute and .sql are alias)
con.sql("""
            SELECT *
            FROM 'data/raw/*.csv'
            WHERE Nome like '%Smirnov'
            """).df()

# %%
# Regex can be used - Case 3
con.execute("""
            FROM 'data/raw/*.csv'
            WHERE regexp_matches(Nome, '^.*Sm.*$')
            """).df()

# %%
# Regex can be used - Case 3 - Regex on column names
con.execute("""
            SELECT columns('N.*')
            FROM 'data/raw/*.csv'
            WHERE regexp_matches(Nome, '^.*Sm.*$')
            """).df()

# %%
# Regex can be used - Case 4 - Regex on column names
duckdb.sql("""
           SELECT columns('shared.*')
           FROM read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2023-03.parquet')
           LIMIT 10;
           """).df()

# %%
# Storing a .csv file in pandas dataframe
df_alemanha = pd.read_csv('data/raw/Alemanha.csv')

# Call the created dataframe in FROM argument
duckdb.sql("""
           SELECT *
           FROM df_alemanha
           """).df()
# %%
# Close connection
con.close()
# %%
