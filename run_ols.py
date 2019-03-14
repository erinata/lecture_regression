from sklearn import linear_model
import pandas as pd

dataset = pd.read_csv("ols_dataset.csv")

print(dataset.head())


target = dataset.iloc[:,2].values

print(target)

data = dataset.iloc[:,3:10]

print(data.head())

regression = linear_model.LinearRegression()

regression.fit(data, target)











