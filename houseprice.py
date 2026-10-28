
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "Area": [1000, 1500, 2000, 2500, 3000],
    "Price": [20, 30, 40, 50, 60]
}

df = pd.DataFrame(data)

X = df[["Area"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

area = [[1800]]
price = model.predict(area)

print("Predicted house price:", round(price[0], 2), "Lakhs")
