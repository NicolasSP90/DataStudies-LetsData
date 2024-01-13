#%%
#%%
# Libraries
import pandas as pd
import PyPDF2
from PyPDF2 import PdfReader
import re
# %%
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
#Pages 7 to 236
relevant_text = extracttext(7, 236, "edital_prograd_2023_370-2.pdf")
relevant_text
#%%
relevant_text = re.sub("Serviço Público Federal", "", relevant_text)
relevant_text = re.sub("Ministério da Educação", "", relevant_text)
relevant_text = re.sub("Fundação Universidade Federal de Mato Grosso do Sul", "", relevant_text)
relevant_text = re.sub("ANEXO I - CANDIDATOS HOMOLOGADOS", "", relevant_text)
relevant_text = re.sub("(Edital nº\xa0370/2023-Prograd/UFMS)", "", relevant_text)
relevant_text = re.sub("INSC. NOME CÓD. CURSO MODALIDADE T.V.LOCAL DE ", "", relevant_text)
relevant_text = re.sub("PROVAL.E.", "", relevant_text)
relevant_text = re.sub(r"\n", r"", relevant_text)
relevant_text = re.sub(r" ", r"", relevant_text)
relevant_text = re.sub(r"Espanhol", r"Espanhol\n", relevant_text)
relevant_text = re.sub(r"Inglês", r"Inglês\n", relevant_text)
relevant_text = re.sub(r"pg.[\d]+", r"\n", relevant_text)
relevant_text
#%%
relevant_text.encode("utf-8")
#%%
relevant_text = relevant_text.split("\n")
# %%
for i in relevant_text:
    if "Legenda" in i:
        relevant_text.remove(i)
    else:
        pass
relevant_text
# %%
df_courses = pd.DataFrame(data=relevant_text)
df_courses.columns = ["full_text"]
# "id", "type", "VL", "VH", "VN", "VM", "RED"
# %%
df_courses.head()
#%%
df_courses.tail()
# %%
df_courses = df_courses[:-1]
df_courses.tail()
# %%
df_courses[df_courses["full_text"] == ""].value_counts()
# %%
df_courses["id"] = df_courses["full_text"].str.extract(r"(\d+)")
df_courses.head()
#%%
df_courses[df_courses["id"] == "963760"]
# %%
df_courses["course"] = df_courses["full_text"].str.extract(r"[\d]+[\w]+[\d]+([\w]+)")
df_courses.head()
# %%
df_courses = df_courses.drop(["full_text"], axis=1)
df_courses.head()
#%%
df_courses.shape
# %%
df_courses.to_csv("./data/df_courses.csv")
# %%
