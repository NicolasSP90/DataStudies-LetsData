#%%
# Importing Libraries
import pandas as pd
import PyPDF2
from PyPDF2 import PdfReader
import re
#%%
# Function to extract text from a PDF file, with a start and end page
def extracttext(firstpage, lastpage, filename):
    pdftext = ""
    
    #Open file
    pdf_file = open(f"./data/{filename}", "rb")

    #Instance the Reader
    pdf_reader = PdfReader(pdf_file)

    # Starting page is 0, not 1
    firstpage = firstpage - 1
    lastpage = lastpage -1
    currentpage = firstpage

    # Add text of the page to the string
    while currentpage <= lastpage:
        pdftext = pdftext + pdf_reader.pages[currentpage].extract_text()
        currentpage += 1

    return pdftext
#%%
# Extracting text from pages 4 to 73
relevant_text = extracttext(4, 73, "edital_prograd_9.pdf")

relevant_text
#%%
# Removing unwanted text
relevant_text = re.sub("Serviço Público Federal", "", relevant_text)
relevant_text = re.sub("Ministério da Educação", "", relevant_text)
relevant_text = re.sub("Fundação Universidade Federal de Mato Grosso do Sul", "", relevant_text)
relevant_text = re.sub("ANEXO I – RESULTADO PRELIMINAR DA PROVA OBJETIVA E REDAÇÃO", "", relevant_text)
relevant_text = re.sub("(Edital nº 9/2024-Prograd/UFMS)", "", relevant_text)
relevant_text = re.sub("INSCRIÇÃO TIPO DE VAGA VL VH VN VM RED", "", relevant_text)
relevant_text
#%%
# Encoding text
relevant_text.encode("utf-8", "ignore")
#%%
# Creating a list from text, separating values at each line break
relevant_text = relevant_text.split("\n")

relevant_text
#%%
# Removing rows with unwanted values
for i in relevant_text:
    if i == r"()":
        relevant_text.remove(i)
    elif "pg." in i:
        relevant_text.remove(i)
    else:
        pass

for i in relevant_text:
    if i == r"()":
        relevant_text.remove(i)
    elif "pg." in i:
        relevant_text.remove(i)
    else:
        pass

relevant_text
#%%
# Dataframe from text list
df_grades = pd.DataFrame(data=relevant_text)
df_grades.columns = ["full_text"]
# "id", "type", "VL", "VH", "VN", "VM", "RED"
#%%
# Head Visualization
df_grades.head()
#%%
# Tail Visualization
df_grades.tail()
#%%
# DataFrame size
df_grades.shape
#%%
# Checking for empty values
df_grades[df_grades["full_text"] == ""].value_counts()
#%%
# Removing textless columns
condition = (df_grades["full_text"] == "")
df_grades = df_grades[~condition]
df_grades.shape
#%%
# Adding id column with values
df_grades["id"] = df_grades["full_text"].str.extract(r"(\d+)")
df_grades.head()
#%%
# Adding type column with values
df_grades["type"] = df_grades["full_text"].str.extract(r"[\d]+\s([\w]+)\s")
df_grades.head()
#%%
# Adding VL grade column with values
df_grades["VL"] = df_grades["full_text"].str.extract(r"[\d]+\s[\w]+\s([\d\,]+)\s")
df_grades.head()
#%%
# Adding VH grade column with values
df_grades["VH"] = df_grades["full_text"].str.extract(r"[\d]+\s[\w]+\s[\d\,]+\s([\d\,]+)\s")
df_grades.head()
#%%
# Adding VN column with values
df_grades["VN"] = df_grades["full_text"].str.extract(r"[\d]+\s[\w]+\s[\d\,]+\s[\d\,]+\s([\d\,]+)\s")
df_grades.head()
#%%
# Adding VM column with values
df_grades["VM"] = df_grades["full_text"].str.extract(r"[\d]+\s[\w]+\s[\d\,]+\s[\d\,]+\s[\d\,]+\s([\d\,]+)\s")
df_grades.head()
#%%
# Adding RED column with values
df_grades["RED"] = df_grades["full_text"].str.extract(r"[\d]+\s[\w]+\s[\d\,]+\s[\d\,]+\s[\d\,]+\s[\d\,]+\s(.*)")
df_grades.head()
#%%
# Dropping original text column and keeping only the created
df_grades = df_grades.drop(["full_text"], axis=1)
df_grades.head()
#%%
# Transforming text to numeric
df_grades["VL"] = df_grades["VL"].str.replace("," , ".")
df_grades["VL"] = df_grades["VL"].astype(float)
vlweight = 1

df_grades["VH"] = df_grades["VH"].str.replace("," , ".")
df_grades["VH"] = df_grades["VH"].astype(float)
vhweight = 1

df_grades["VN"] = df_grades["VN"].str.replace("," , ".")
df_grades["VN"] = df_grades["VN"].astype(float)
vnweight = 1

df_grades["VM"] = df_grades["VM"].str.replace("," , ".")
df_grades["VM"] = df_grades["VM"].astype(float)
vmweight = 1

df_grades["RED"] = df_grades["RED"].str.replace("," , ".")
df_grades["RED"] = df_grades["RED"].astype(float)
redweight = 2
# %%
df_grades["Total"] = ((
    df_grades["VL"]*vlweight + 
    df_grades["VH"]*vhweight + 
    df_grades["VN"]*vnweight + 
    df_grades["VM"]*vmweight + redweight*df_grades["RED"])
    /
    (vlweight + vhweight + vnweight + vmweight + redweight))
# %%
df_grades.info()
#%%
# Head Visualization
df_grades.head()
# %%
# Saving to CSV
df_grades.to_csv("./data/df_grades.csv")
# %%
