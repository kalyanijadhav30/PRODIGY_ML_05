from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset (acts as food images)
digits = datasets.load_digits()

X = digits.data
y = digits.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = SVC()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, predictions))

# Simulated food labels
food_map = {
    0: ("Apple", 52),
    1: ("Burger", 295),
    2: ("Pizza", 266),
    3: ("Rice", 130),
    4: ("Banana", 89),
    5: ("Sandwich", 250),
    6: ("Pasta", 131),
    7: ("Fries", 312),
    8: ("Salad", 150),
    9: ("Ice Cream", 207)
}

# Show sample predictions
print("\nSample Predictions:")
for i in range(5):
    pred = predictions[i]
    food, calories = food_map[pred]
    print(f"Predicted Food: {food}, Estimated Calories: {calories} kcal")