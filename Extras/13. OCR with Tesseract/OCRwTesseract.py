# %%
# OCR (or Optical Character Recognition), is the ability to extract text from images.
# This exercise will use Tesseract to extract texts from images.
# Tesseract is a open source project
# %%
# Options to work with Tesseract
!tesseract --help-extra
# %%
# Import Libraries
import cv2
from matplotlib import pyplot as plt
import pytesseract
import itertools
import os
import string
# %%
# Reading image
image = cv2.imread(r"./data/plate/62417_1000x600_width.jpg")

# Showing image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
# %%
# Extracting images
pytesseract.image_to_string(image)
# %%
# List of available languages
!tesseract --list-langs

# If needed, donwload prétrained model and put the file in Tesseract-OCR/tessdata
# %%
# Extracting images in pt-br
pytesseract.image_to_string(image, lang="por")

# %%
# Reading image
image = cv2.imread(r"./data/plate/placa-eunicio.jpg")

# Showing image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
# %%
# Extracting images
pytesseract.image_to_string(image, lang="por")
# %%
# Extra configurations
pytesseract.image_to_string(image, lang="por", config="--psm 6 --oem 1")
# %%
# Trying all configurations
list_psm = set(range(14))
list_oem = set(range(4))

combinations = list(itertools.product(list_psm, list_oem))

for psm, oem in combinations:
    try:
        print(f"\n\Combinaton: PSM = {psm} ; OEM = {oem}")
        result = pytesseract.image_to_string(image, lang="por", config = f"--psm {psm} --oem {oem}")
        print(result)
    except:
        print("Something Went Wrong!")
# %%
# Running the best configuration
pytesseract.image_to_string(image, lang="por", config = "--psm 4 --oem 1")
# %%
# Creating function to show image
def showIMG(imgpath):
    # Reading image
    image = cv2.imread(imgpath)

    # Showing image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
# %%
# Creating function to extract text    
def extract_text(imgpath, language=None, psm=None, oem=None, char=None):
    # Reading image
    image = cv2.imread(imgpath)

    # Setting arguments
    arguments = {"image" : imgpath}

    if language is not None:
        arguments["lang"] = language

    if psm is not None:
        psm_arg = f"--psm {psm}"
    else:
        psm_arg = f""

    if oem is not None:
        oem_arg = f"--oem {oem}"
    else:
        oem_arg = f""
    
    if char is not None:
        char_arg = f"-c tessedit_char_whitelist={char}"
    else:
        oem_arg = f""

    arguments["config"] = f"{psm_arg} {oem_arg} {char_arg}"
    
    # Extracting Text
    text = pytesseract.image_to_string(**arguments)
    
    return text
# %%
# list of images in the folder
plates_path = r"./data/plate/"

images_plates = os.listdir(plates_path)

images_plates

# %%
# Extracting for all imagens in the plate folder
for image in images_plates:
    file = os.path.join(plates_path, image)

    showIMG(file)

    text = extract_text(file, language="por", psm=4, oem=1)

    print(text)
# %%
# Creating a list of car plates
car_path = "./data/cars/"

files_cars = os.listdir(car_path)

files_cars
# %%
# Extracting text
for img_car in files_cars:
    img_car_plate = os.path.join(car_path, img_car)
    showIMG(img_car_plate)
    print(extract_text(img_car_plate, psm=4, oem=1))

# %%
# Setting a car plate to create improve detection
img_imp = os.path.join(car_path, files_cars[1])
showIMG(img_imp)
# %%
# Finding psm and oem parameters
for psm, oem in combinations:
    try:
        print(f"\n\nCombination: psm {psm} oem {oem}")
        result = pytesseract.image_to_string(img_imp, config=f"--psm {psm} --oem {oem}")
        print(result)
    except:
        print("Error")
# %%
# Creating a list of valid characters
char_list = string.ascii_uppercase + string.digits
char_list

# %%
for img_car in files_cars:
    img_car_plate = os.path.join(car_path, img_car)
    showIMG(img_car_plate)
    print(extract_text(img_car_plate, psm=11, oem=3, char=char_list))
# %%
