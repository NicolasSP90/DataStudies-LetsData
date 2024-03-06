# %%
# importing libraries
import duckdb
import pandas as pd
import os

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
