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

X = [
	[24,55,31,3,0,7,20],
	[40,50,2,5,1,8,20],
	[3,95,37,3,1,15,17],
]

results = regression.predict(X)

print(results)









