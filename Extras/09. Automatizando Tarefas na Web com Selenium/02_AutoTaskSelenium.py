#%%
# Importing Libraries
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

#%%
link = 'https://www.ibge.gov.br/pt/inicio.html'

# Web Driver
driver = webdriver.Edge() # Microsoft Edge
# driver = webdriver.Chrome() # Google Chrome
# driver = webdriver.Firefox() # Firefox

# Acessing Website
driver.get(link)

#%%
# Setting Wait Time
wait_time = WebDriverWait(driver,15)

# Loading Page with Waiting Time
loaded_page = wait_time.until(EC.title_contains("IBGE"))

#%%
# Checking if Page is Loaded
if loaded_page:
    btn_indicators = driver.find_element(By.CLASS_NAME, 'btn-outros-indicadores')
else:
    print("Page not Loaded")

#%%
# Scrolling
driver.execute_script("window.scrollBy(0,250)")

#%%
# Clicking in the Button
btn_indicators.send_keys(Keys.RETURN)


#%%
wait_time = WebDriverWait(driver, 5)

#%%
# Checking if is the right page
assert "Painel de Indicadores" in driver.title

#%%
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
df_indicators.iloc[1:20:2]# %%

# %%
