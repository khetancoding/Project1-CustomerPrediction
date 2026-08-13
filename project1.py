import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Temporary dataset
data = {
    "Customer_ID": [1001, 1002, 1003, 1004, 1005],
    "Age": [25, 41, -3, 33, 29],
    "Annual_Income": [55000, 82000, 70000, 91000, 40000],
    "Purchases": [5, 12, 7, 18, 2],
    "Total_Spent": [200, 1400, 650, 2500, 80],
    "Cart_Value": [40, 120, 0, 300, 15],
    "Items_In_Cart": [2, 5, 0, 6, 1],
    "Time_On_Site": [15, 32, 2, 45, 8],
    "Bought_Again": ["Yes", "Yes", "No", "Yes", "No"]
}

df = pd.DataFrame(data)

# Remove customers with impossible ages
df = df[df["Age"] >= 0]

print(df)
print(df.head())
print(df.info())
print(df.describe())
print(df[["Age", "Purchases"]])

df["Average_Spend_Per_Purchase"] = df["Total_Spent"] / df["Purchases"]
print(df["Average_Spend_Per_Purchase"])

df["High_Value_Customer"] = df["Total_Spent"] > 1000
print(df[df["High_Value_Customer"]])
print(df["High_Value_Customer"])

df["Big_Shopper"] = df["Purchases"] > 10
print(df[df["Big_Shopper"]])
print(df["Big_Shopper"])

df["Customers_over_30"] = df["Age"] > 30
print(df[df["Customers_over_30"]])
print(df["Customers_over_30"])

df["Older30&Spent_over$1000"] = (df["Age"] > 30) & (df["Total_Spent"] > 1000)
print(df["Older30&Spent_over$1000"])

X = df[
        [
            "Age",
            "Annual_Income",
            "Purchases",
            "Total_Spent",
            "Cart_Value",
            "Items_In_Cart",
            "Time_On_Site",
            "Average_Spend_Per_Purchase",
            "High_Value_Customer",
            "Big_Shopper",
            "Customers_over_30",
            "Older30&Spent_over$1000"
        ]
    ]
y = df["Bought_Again"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Features:")
print(X_train)

print("Testing Features:")
print(X_test)

print("Training Labels:")
print(y_train)

print("Testing Labels:")
print(y_test)

model = LogisticRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Predictions:")
print(predictions)

print("Actual:")
print(y_test)