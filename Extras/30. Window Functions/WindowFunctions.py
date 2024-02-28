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
        print(f"Connected with Database. sqlite version {sqlite3.sqlite_version}")
    
    except Error as e:
        print(f"Error {e} in the connection")
    
    return db_con

# %%
# Function to create table
def create_table(db_con):
    try:
        query = """
                    CREATE TABLE table_sales (
                        ID INTEGER PRIMARY KEY,
                        Vendor TEXT NOT NULL,
                        Product TEXT NOT NULL,
                        Quantity TEXT NOT NULL,
                        Sales_Date DATE NOT NULL
                    );
                """
        
        db_con.execute(query)
        print("Table was successfully created!")
    
    except Error as e:
        print(f"Error {e}  in the table creation")

# %%
# Function to insert data
def insert_data(db_con):
    try:
        query = """
                    INSERT INTO vendas(ID, Vendor, Product, Quantity, Sales_Date) VALUES
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
        
        db_con.execute(query)
        db_con.commit()
        print("Successfully insert Data")

    except Error as e:
        print(f"Error {e} when inserting data")

# %%
# Using the functions to connect, create base anr 
db_connection = db_connect()

with db_connection:
    create_table(db_connection)
    insert_data(db_connection)
# %%
db_connection.close()

# %%