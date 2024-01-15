#%%
# Importing Libraries
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import  WebDriverWait
from selenium.webdriver.support import expected_conditionsa 
#%%
# Web Driver
driver = webdriver.Edge() # Microsoft Edge
# driver = webdriver.Chrome() # Google Chrome
# driver = webdriver.Firefox() # Firefox
driver.get("http://letsdata.ai")
# %%
