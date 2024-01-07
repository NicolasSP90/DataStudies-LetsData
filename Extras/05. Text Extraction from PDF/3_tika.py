#%%
# Importing Libraries
from tika import parser
# %%
pdf_reader = parser.from_file("./data/pdf.pdf")
# For this case it wont work unless Java is installed
text = pdf_reader["content"]
text
# %%
