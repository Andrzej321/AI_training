import pandas as pd

df = pd.read_csv("hyperparams.csv", delimiter=";")

# Show the DataFrame
print(df["sequence_size"][2])