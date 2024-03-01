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

