# %%
# Considering the Following
list_even_square_1 = []
for i in range(20):
    if i% 2 ==0:
        list_even_square_1.append(i**2)
print(list_even_square_1)

# using List Comprehension:
list_even_square_2 = [i**2 for i in range(20) if i%2 == 0]
print(list_even_square_2)
# %%
# Working with Tuples
[(i,i+1) for i in range(10)]
# %%
# Working with names
list_names = ["Bernardo Lago", "Felipe Schiavon", "Leon Solon"]
[name.split()[0] for name in list_names]
# %%
# Strings and conditions
list_food = ["barbecue", "shrimp", "watermellon", "pizza"]
[food for food in list_food if food.endswith("za")]
# %%
# Several iterations
list_know = ["Python", "Pandas", "Scikit Learn"]

[(name, know) for name in list_names for know in list_know]
# %%
# Several conditions
[(name, know) for name in list_names if not (name.startswith("F")) for know in list_know if know.startswith("P")]
# %%
# Dictionary from 2 lists
list_price = [42.8, 55.3, 10.5, 40.8]
dict_food = {food:price for food, price in zip(list_food, list_price)}
dict_food
# %%
# Dictionary from 1 list
dict_names = {name:position for position, name in enumerate(list_names)}
dict_names
# %%
# Set Comprehension
{element for element in [1,1,3,3,3,3,4,5]}
# %%
