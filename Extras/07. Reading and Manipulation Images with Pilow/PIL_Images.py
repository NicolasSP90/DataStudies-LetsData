#%%
# PIL Libraries
from PIL import Image
from PIL import ImageFilter

# Path manipulation
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
    jpg_file.save(f"{folder_save}converted_{filename}.png")
# %%
converIMG(list_files[0])
# %%
# Creating thumbnails
def createThumb(filename):
    folder_save = r"./data/thumb/"
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    thumb_size = (150, 150)
    thumb_file = Image.open(f"{folder_dir}/{filename}")
    thumb_file.thumbnail(thumb_size)
    thumb_file.save(f"{folder_save}thumb_{filename}")
    return thumb_file
# %%
createThumb(list_files[3])
# %%
def rotateIMG(filename, angle):
    folder_save = r"./data/rotate/"
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    rotate_file = Image.open(f"{folder_dir}/{filename}")
    rotate_file = rotate_file.rotate(angle)
    rotate_file.save(f"{folder_save}rotate_{filename}")
    return rotate_file
# %%
rotateIMG(list_files[4], 90)
# %%
# Converting to grayscale
def grayscaleIMG(filename):
    folder_save = r"./data/grayscale/"
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    gray_file = Image.open(f"{folder_dir}/{filename}")
    gray_file = gray_file.convert("L")
    gray_file.save(f"{folder_save}grayscale_{filename}")
    return gray_file
# %%
grayscaleIMG(list_files[5])
# %%
# Using Blur filter
def filterBlurIMG(filename, blur_intensity):
    folder_save = r"./data/filter/"
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    filter_file = Image.open(f"{folder_dir}/{filename}")
    filter_file = filter_file.filter(ImageFilter.GaussianBlur(blur_intensity))
    filter_file.save(f"{folder_save}blur_{blur_intensity}_{filename}")
    return filter_file
# %%
filterBlurIMG(list_files[6],5)
# %%
def filterContourIMG(filename):
    folder_save = r"./data/filter/"
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    filter_file = Image.open(f"{folder_dir}/{filename}")
    filter_file = filter_file.filter(ImageFilter.CONTOUR)
    filter_file.save(f"{folder_save}contour_{filename}")
    return filter_file
# %%
filterContourIMG(list_files[6])
# %%
# Grouping images
def groupIMG(filename1, filename2, pixels=(100,100), position=(0,0)):
    # Image Name
    pixel1, pixel2 = pixels
    final_name = f"{list_files.index(filename1)}_{list_files.index(filename2)}__{pixel1}x{pixel2}"
    imgformat = ".jpeg"
    
    # Final Folder
    folder_save = r"./data/grouping/"
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    
    # Loading Images
    group_img1 = Image.open(f"{folder_dir}/{filename1}")
    group_img2 = Image.open(f"{folder_dir}/{filename2}")

    # Position of image 2 inside image 1 (in %)
    img_w, img_h = group_img1

    group_img2.thumbnail(pixels)
    group_img1.paste(group_img2, position)
    group_img1.save(f"{folder_save}grouping_{final_name}{imgformat}")
    return group_img1
# %%
groupIMG(list_files[0], list_files[1], (200,200), )
# %%
list_files.index(list_files[2])
# %%
