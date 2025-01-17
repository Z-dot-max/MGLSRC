import pandas as pd

df = pd.read_csv("../data/filtered_USPTO_condition_sum.csv")

with open("../data/smiles.txt", "w") as f:
    for _, row in df.iterrows():
        reactants = eval(row['Reactants'])
        for reactant in reactants:
            f.write(reactant + "\n")
        
        products = eval(row['Products'])
        for product in products:
            f.write(product + "\n")
