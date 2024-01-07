#%%
# Libraries
import camelot
# %%
pdf_table = camelot.read_pdf("./data/foo.pdf")
pdf_table
# %%
pdf_table[0]
# %%
df_tabela = pdf_table[0].df
df_tabela

# %%
