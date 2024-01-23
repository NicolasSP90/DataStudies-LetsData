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
