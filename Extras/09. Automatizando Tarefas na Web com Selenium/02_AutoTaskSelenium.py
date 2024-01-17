# %%
# Importing Libraries
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# %%
# Acessing Website
link = 'https://www.ibge.gov.br/pt/inicio.html'

# Web Driver
driver = webdriver.Edge() # Microsoft Edge
# driver = webdriver.Chrome() # Google Chrome
# driver = webdriver.Firefox() # Firefox

driver.get(link)

# %%
# Setting Wait Time
wait_time = WebDriverWait(driver,15)

# Loading Page with Waiting Time
loaded_page = wait_time.until(EC.title_contains("IBGE"))

# %%
# Checking if Page is Loaded
if loaded_page:
    btn_indicators = driver.find_element(By.CLASS_NAME, 'btn-outros-indicadores')
else:
    print("Page not Loaded")

# %%
# Scrolling
driver.execute_script("window.scrollBy(0,250)")

# %%
# Clicking in the Button
btn_indicators.send_keys(Keys.RETURN)

# %%
wait_time = WebDriverWait(driver, 5)

# %%
# Checking if is the right page
assert "Painel de Indicadores" in driver.title

# %%
# Importing HTML
table_indicators = driver.find_element(By.CLASS_NAME, "indicadores-tabela")

# HTML table
html_table = table_indicators.get_attribute('outerHTML')
html_table

# %%
# Importing to Pandas
dataframe_HTML = pd.read_html(html_table)
len(html_table)
# %%
# Indicators Dataframe
df_indicators = dataframe_HTML[0]
df_indicators

# %%
# Checking for the numeric data
df_indicators.iloc[1:20:2]

# %%
# Checking the index
df_indicators.iloc[1:20:2].index

# %%
# Dropping unwanted data
df_indicators = df_indicators.drop(df_indicators.iloc[1:20:2].index).reset_index(drop=True)
df_indicators

# %%
# Checking for IPCA15 (getting all classes with the same name)
non_sprite = driver.find_elements(By.CLASS_NAME, "nonsprite")
len(non_sprite)

# %%
# Checking for IPCA15
for element in non_sprite:
    if "IPCA-15" in element.get_attribute("innerHTML"):
        break
element.get_attribute("innerHTML")

# %%
# Clicking in the IPCA15
element.click()

# %%
# Looking for Sidra's link with data
# In this case, it's only one element
link_sidra = driver.find_element(By.PARTIAL_LINK_TEXT, "Sidra - Tabelas de resultados")
link_sidra

# %%
# Clicking in Sidra's link
link_sidra.click()

# %%
# Checking if it is the right page
assert "SIDRA" in driver.title

# %%
# Checking for downloadable data
arrow = driver.find_element(By.CLASS_NAME, "glyphicon-download")
arrow

# %%
# Clicking in the data to be downloaded
arrow.click()

# %%
# Download as XLSX
link_xlsx = driver.find_element(By.XPATH, "//a[@title='Exportar em XLSX']")
link_xlsx.click()

# %%
# Return to the previous page
driver.back()

# %%
# Textual search in the website
search_box = driver.find_element(By.NAME, 'searchword')

# Insert IGPM in the serch bar
search_box.send_keys("IGPM")

# Pressing enter
search_box.send_keys(Keys.RETURN)
