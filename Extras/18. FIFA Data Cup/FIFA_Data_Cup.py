# Importing Libraies
import cv2
from easyocr import Reader
import pandas as pd

# Capturing videos in streaming
vid = cv2.BideoCapture(1)

# Updating text to show the user
text = ""
first_name = ""
last_name = ""
club = ""
height = ""
weight = ""
value = ""

# Importing data
df_fifa = pd.read_csv("./data/FIFA23_official_data.csv")

# Function to get the statistics + 1st and last name
def player_search(first_name, last_name):
    name_search = f"{first_name[0]}. {last_name}"
    name_search = name_search.lower()

    # Search for statistics
    player_stats = df_fifa[df_fifa["Name"].str.lower() == name_search, ["Club", "Height", "Weight", "" ]]




