# %%
# Importing libraries
import pandas as pd
import sqlite3
from sqlite3 import Error

# %%
# Function to create a connection with the database
def db_connect():
    db_con = None

    try:
        db_con = sqlite3.connect("sales.db")
        print(f"Connected with Database. sqlite version {sqlite3.sqlite_version}.")
    
    except Error as e:
        print(f"Error {e} in the connection.")
    
    return db_con

# %%
# Define as a global variable
db_connection = db_connect()

# Closing the connection
db_connection.close()

# %%
# Function to create table
def create_table(query):
    db_connection = db_connect()
    with db_connection:
        try:
            db_connection.execute(query)
            print("Table was successfully created!")

        except Error as e:
            print(f"Error {e}  in the table creation.")

# %%
# Function to insert data
def insert_data(query):
    db_connection = db_connect()
    with db_connection:
        try:
            db_connection.execute(query)
            db_connection.commit()
            print("Successfully insert Data.")

        except Error as e:
            print(f"Error {e} when inserting data.")

# %%
# Using the functions to create database
# Query to create table:
query = """
        CREATE TABLE table_sales (
            ID INTEGER PRIMARY KEY,
            Vendor TEXT NOT NULL,
            Product TEXT NOT NULL,
            Quantity TEXT NOT NULL,
            Sales_Date DATE NOT NULL
            );
        """
    
create_table(query)

# %%
# Using the functions to insert data
# Query to insert data
query = """
        INSERT INTO table_sales (ID, Vendor, Product, Quantity, Sales_Date) VALUES
            (1, 'Ana', 'Book', 10, '2023-01-01'),
            (2, 'Beto', 'Pencil', 30, '2023-01-05'),
            (3, 'Carlos', 'Notebook', 15, '2023-01-08'),
            (4, 'Ana', 'Notebook', 20, '2023-01-09'),
            (5, 'Beto', 'Pen', 50, '2023-01-12'),
            (6, 'Carlos', 'Book', 25, '2023-01-15'),
            (7, 'Ana', 'Pen', 30, '2023-01-17'),
            (8, 'Beto', 'Notebook', 40, '2023-01-19'),
            (9, 'Carlos', 'Pencil', 35, '2023-01-22'),
            (10, 'Ana', 'Book', 45, '2023-01-25');
        """
        
insert_data(query)

# %%
# Function to execute query
def exe_query(query):
    db_connection = db_connect()
    with db_connection:
        try:
            df = pd.read_sql(query, db_connection)
            display(df)

        except Error as e:
            print(f"Error {e} when trying to execute the query.")

# %%
# Query: all rows
query = """
        SELECT *
        FROM table_sales;
        """

# Execute query
exe_query(query)

# %%
# Query: cout all sales
query = """
        SELECT Vendor, Product, Quantity,
        SUM(Quantity) OVER() as Total_Salaes
        FROM table_sales;
        """

# Execute query
exe_query(query)

# %%
# Query: average sales
query = """
        SELECT Vendor, Product, Quantity,
        AVG(Quantity) OVER() as Average_Salaes
        FROM table_sales;
        """

# Execute query
exe_query(query)

# %%
# Query: Mmx sales
query = """
        SELECT Vendor, Product, Quantity,
        MAX(Quantity) OVER() as Maximum_Qnt
        FROM table_sales;
        """

# Execute query
exe_query(query)

# %%
# Query: Sum and partition by vendor
query = """
        SELECT Vendor, Product, Quantity,
        SUM(Quantity) OVER(PARTITION BY Vendor) as Qnt_by_Vendor
        FROM table_sales;
        """

# Execute query
exe_query(query)

# %%
# Query: Average and partition by vendor
query = """
        SELECT Vendor, Product, Quantity,
        AVG(Quantity) OVER(PARTITION BY Vendor) as Mean_by_Vendor
        FROM table_sales;
        """

# Execute query
exe_query(query)

# %%
# Query: ranking of greatest sales to lowest
query = """
        SELECT Vendor, Product, Quantity,
        RANK() OVER (ORDER BY Quantity DESC) AS Ranking
        FROM table_sales;
        """

# Execute query
exe_query(query)

# %%
# Query: ranking of greatest sales to lowest (without draws)
query = """
        SELECT Vendor, Product, Quantity,
        DENSE_RANK() OVER (ORDER BY Quantity DESC) AS Ranking
        FROM table_sales;
        """

# Execute query
exe_query(query)

# %%
# Query: ranking of greatest sales to lowers, partitioned by vendor
query = """
        SELECT Vendor, Product, Quantity,
        RANK() OVER (PARTITION By Vendor ORDER BY Quantity DESC) as Ranking
        FROM table_sales;
        """

# Execute query
exe_query(query)

# %%
# Query: Regular GROUP BY vendor
query = """
        SELECT Vendor, SUM(Quantity)
        FROM table_sales
        GROUP BY Vendor;
        """

# Execute query
exe_query(query)

# %%
# Query: ranking of vendors by quantity
query = """
        SELECT Vendor, SUM(Quantity),
        RANK() OVER (ORDER BY SUM(Quantity) DESC) AS Ranking
        FROM table_sales
        GROUP BY Vendor;
        """

# Execute query
exe_query(query)

# %%
# Query: LAG function
query = """
        SELECT Vendor, Product, Quantity, Sales_Date,
        LAG (Quantity) OVER (ORDER BY Sales_Date) AS Prev_Qnt,
        Quantity / LAG(Quantity) OVER (ORDER BY Sales_Date) AS Stats
        FROM table_sales;
        """

# Execute query
exe_query(query)

# %%
# Query: LAG of Quantity, Partition by Vendor and Order by Sales_Date
query = """
        SELECT Vendor, Product, Quantity, Sales_Date,
        LAG (Quantity) OVER (PARTITION BY Vendor ORDER BY Sales_Date)
        FROM table_sales;
        """

# Execute query
exe_query(query)# %%

# %%
# Query: LEAD of Quantity, Partition by Vendor
query = """
        SELECT Vendor, Product, Quantity, Sales_Date,
        LEAD(Quantity) OVER (PARTITION BY Vendor) AS Next_Qnt
        FROM table_sales;
        """

# Execute query
exe_query(query)

# %%
# Query: LEAD of Quantity, Partition by Vendor and Order by Sales_Date
query = """
        SELECT Vendor, Product, Quantity, Sales_Date,
        LEAD(Quantity) OVER (PARTITION BY Vendor ORDER BY Sales_Date) AS Next_Qnt
        FROM table_sales;
        """

# Execute query
exe_query(query)
# %%
