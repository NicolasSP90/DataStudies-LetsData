# %%
# Importing Libraries
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# %%
# Products with Labels
products = [
    ("Smartphone", ["Electronic", "Communication", "Portable"]),
    ("Notebook Gamer", ["Electronic", "Game", "Portable", "Computer"]),
    ("Running Shoes", ["Clothing", "Sports", "Footwear"]),
    ("Blender", ["Appliances", "Kitchen", "Food"]),
    ("Mountain Bike", ["Sports"]),
    ("Cookbook", ["Books", "Kitchen", "Education"])
    ]

products

# %%
# Creating a list of labels
labels = [product[1] for product in products]

labels

# %%
# Instancing MultiLabelBinarizer
mlb = MultiLabelBinarizer()

binarizer_labels = mlb.fit_transform(labels)

# %%
# Checking the atributes of mlb and the binarized labels
display(len(mlb.classes_))
print(mlb.classes_)
print("\n")
print(binarizer_labels)
print("\n")
print(type(binarizer_labels))

# %%
# Printing results
for product, bin_label in zip(products, binarizer_labels):
    print(f"Product: {product[0]}")
    print(f"Original Labels: {product[1]}")
    print(f"Binarized Labels: {bin_label}\n")

# %%
# Reverting the Binarized Labels
reverted_labels = mlb.inverse_transform(binarizer_labels)
reverted_labels

# %%
# Using Pandas
df = pd.DataFrame(data=products, columns=["products", "labels"])
df

# %%
# Binarizer applied to a dataframe column
mlb = MultiLabelBinarizer()

binarizer_labels = mlb.fit_transform(df["labels"])
display(binarizer_labels)

# Storing the resulting binarizer labels in a dataframe
df_binarizer = pd.DataFrame(data=binarizer_labels, columns=mlb.classes_)
display(df_binarizer)

# Merging Dataframes
# Note: both concat and merge can be used. Usually concat is used for homogeneous dataframes and merge is used for complementary dataframes
df_final = pd.merge(left= df, 
                    right= df_binarizer, 
                    how= "left",
                    left_index=True,
                    right_index=True
                    )

display(df_final)

# Dropping unused columns
df_final = df_final.drop(columns=["labels"], axis="columns")

display(df_final)

# %%
