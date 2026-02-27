import pandas as pd

df = pd.read_csv("datasets/final/stylometry_dataset.csv")
print(df["book"].unique())
