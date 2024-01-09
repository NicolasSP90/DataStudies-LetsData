#%%
# Importing Libraries
from PIL import Image
import os

#%%
folder_dir = r"./data/raw"
list_files = os.listdir(folder_dir)
list_files
# %%
# Opening Image
Image.open(f"{folder_dir}/{list_files[0]}")
# %%
Image.open(f"{folder_dir}/{list_files[2]}")
#%%
# Converting image
def converIMG(filename):
    folder_save = r"./data/converted/"
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    jpg_file = Image.open(f"{folder_dir}/{list_files[0]}")
    jpg_file.save(f"{folder_save}dog.png")

converIMG(list_files[0])
# %%
# Creating thumbnails
def createThumb(filename)
    folder_save = r"./data/thumb/"
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    thumb_size = (150, 150)
    thumb_file = 
    Image.open(f"{folder_dir}/{list_files[0]}")
    thumb_file.save(f"{folder_save}dog.png")