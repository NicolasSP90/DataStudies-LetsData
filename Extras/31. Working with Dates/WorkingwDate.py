# %%
# Importing Libraries
import pandas as pd
from datetime import date, time, datetime, timedelta

# %%
# Date Creation - date
date_python = date(2023, 7, 25) #YYYY, MM, DD

display(date_python)
print(date_python)

# %%
# Date Creation - datetime
datetime_python = datetime(2023, 7, 25, 15, 30, 40) #YYYY, MM, DD, HH, MM, SS)

display(datetime_python)
print(datetime_python)

# %%
# Date Creation - time
time_python = time(15,30,40) #HH, MM, SS

display(time_python)
print(time_python)

print(time_python.strftime("%H:%MM"))

# %%
# Date Creation - String
date_time_str = "2023-07-25 15:30:40" #YYYY-MM-DD HH-MM-SS

string_python = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")

display(string_python)
print(string_python)

# %%
# Pandas - TImestamp
date_pandas = pd.Timestamp("2023-07-25T15:30:40") # String ISO 8601

display(date_pandas)
print(date_pandas)

# %%
# Time to String
time_hour = time(15,30,40)

time_hour_string = time_hour.strftime("%H:%M:%S")

print(f"Hora como string: {time_hour_string}")

# %%
# String to Timestamp
date_string = "2023-07-25T15:30:40"
date_string_timestamp = pd.to_datetime(date_string)

print(type(date_string_timestamp))

print(f"Converted from ISO 8601 String: {date_string_timestamp}")

# %%
# Creating a dataframe with date columns
df = pd.DataFrame({"date_string": ["2023-07-25T15:30:40", 
                                  "2023-08-10T16:40:50", 
                                  "2024-09-15T17:50:55"]})

df.info()

# %%
# Creating a new column
df["date"] = pd.to_datetime(df["date_string"])

df.info()

# %%
# Checking dataframe
df

# %%
# Checking the first 4 elements of the data_string column to return the year
df['date_string'].str[:4]

# %%
# Getting the year in the datetime column
df["date"].dt.year

# %%
display(datetime_python)
print(datetime_python)
year = datetime_python.year
month = datetime_python.month
day = datetime_python.day
hour = datetime_python.hour
mins = datetime_python.minute
sec = datetime_python.second

print(f"Year: {year}\nMonth: {month}\nDay: {day}\nHout: {hour}\nMinutes: {mins}\nSeconds: {sec}")

# %%
# Adding yead, month and day columns
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

df

# %%
# Calculating the difference between dates
date1 = datetime(2024, 10, 15)
date2 = datetime(2024, 8, 20)

print(date1)
print(date2)
print(date1 - date2)

# %%
# Formatting Dates
print(date_python)
print(date_python.isoformat())

print(date_pandas)
print(date_pandas.isoformat())

difference = pd.Timedelta(days=5) + pd.Timedelta(days=3)

print(f"Difference in days:{difference.days}")
print(f"Difference in seconds:{difference.total_seconds()}")
print(f"Difference in hours:{(difference.total_seconds())/(60*60)}")

# %%
# Show the dataframe
df

# %%
# Adding columns with a time skip
df["date_5days"] = df["date"] + pd.Timedelta(days=5)
df["date_5years"] = df["date"] + pd.Timedelta(days=5*365)

df

# %%

df["difference_5d"] = df["date_5days"] - df["date"]
df["difference_5y"] = df["date_5years"] - df["date"]

df

# %%
# Checking column difference in seconds
df["difference_5d"].dt.total_seconds()
# %%
