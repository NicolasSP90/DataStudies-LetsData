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

# OCR reader
reader = Reader(["en"])

# Importing data
df_fifa = pd.read_csv("./data/FIFA23_official_data.csv")

# Function to get the statistics + 1st and last name
def player_search(first_name, last_name):
    name_search = f"{first_name[0]}. {last_name}"
    name_search = name_search.lower()

    # Search for statistics
    player_stats = df_fifa[df_fifa["Name"].str.lower() == name_search, ["Club", "Height", "Weight", "" ]]

    # Some name formats differ from each other
    if len(player_stats) == 0:
        name_search = f"{first_name} {last_name}"
        name_search = name_search.lower()
        player_stats = (df_fifa.loc[df_fifa["Name"].str.lower() == name_search, ["Club", "Height", "Weight", "Value"]])
    
    final_stats = player_stats.values[0]

    return final_stats


# Keep streaming while "q" is not pressed
while (True):

    # Capture streaming, frame by frame
    ref, frame = vid.read()

    # Pressing "d" makes the OCR detect the player name on the card
    if cv2.waitKey(1) & 0xFF == ord("d"):
        # OCR
        ocr_result = reader.readtext(frame)
        
        # Assembly text
        for result in ocr_result:
            print(result[1])
            if len(result[1].split()) == 2:
                first_name = result[1].split()[0]
                last_name = result[1].split()[1]

                text = result[1]


    # Once the name is detected, pressing "s" searches for the statistics of the player
    if cv2.waitKey(1) & 0xFF == ord("s"):
        try:
            print(f"Name: {first_name} {last_name}")
            stats_str = player_search(first_name, last_name)
            club = f"Club: {stats_str[0]}"
            height = f"Altura: {stats_str[1]}"
            weight = f"Weight: {stats_str[2]}"
            value = f"Value: {stats_str[3][1:]} Euros"
        
        except Exception as e:
            print(e)
    
    # Clicking "a", erase all data
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Write all text in the emage
    cv2.putText(frame, text, (200, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.putText(frame, club, (200, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.putText(frame, height, (200, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.putText(frame, weight, (200, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.putText(grame, value, (200, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # OpenCV Shows the frame
    cv2.imshow("frame", frame)

# After the loop, closing the object of streaming
vid.release()

# Closing all windows
cv2.destroyAllWindows





