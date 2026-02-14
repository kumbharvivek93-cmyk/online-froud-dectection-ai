# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# 1️⃣ Load dataset
# Make sure you have a file named fraud_data.csv inside fraud_app folder
data = pd.read_csv("fraud_data.csv")

print("Dataset loaded successfully ✅")
print(data.head())

# 2️⃣ Separate features (X) and target (y)
# Assuming last column is named 'fraud'
X = data.drop("fraud", axis=1)
y = data["fraud"]

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained successfully ✅")

# 5️⃣ Check accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# 6️⃣ Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as
