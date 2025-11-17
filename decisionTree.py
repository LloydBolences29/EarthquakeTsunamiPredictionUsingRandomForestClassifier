import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv('earthquake_data_tsunami.csv')

#features and labels
x = data[["latitude", "longitude", "depth", "magnitude", "cdi", "mmi", "sig" ]]
#target variable
y = data["tsunami"]

#splitting of the datasets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


#for decision tree
#model creation
# model = DecisionTreeClassifier() model.fit(x-train, y-train) 

# #testing the model 
# accuracy = model.score(x-test, y-test) 
# print(f"Model Accuracy: {accuracy * 100:.2f}%")


#model creation
model = RandomForestClassifier(
    n_estimators=200,      # more trees
    max_depth=8,           # limit tree depth
    min_samples_split=5,   # require at least 5 samples to split
    random_state=42,
    class_weight='balanced'
)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")


#ask for user input

def get_user_input():
    print("Enter earthquake details to predict tsunami likelihood:")
    latitude = float(input("Latitude: "))
    longitude = float(input("Longitude: "))
    depth = float(input("Depth (km): "))
    magnitude = float(input("Magnitude: "))
    cdi = float(input("Community Determined Intensity (CDI): "))
    mmi = float(input("Modified Mercalli Intensity (MMI): "))
    sig = float(input("Significance (sig): "))

    user_data = pd.DataFrame({
        "latitude": [latitude],
        "longitude": [longitude],
        "depth": [depth],
        "magnitude": [magnitude],
        "cdi": [cdi],
        "mmi": [mmi],
        "sig": [sig]
    })
    return user_data


# Get user input and predict
user_input = get_user_input()
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Show results
print("\n--- Prediction Result ---")
if prediction[0] == 1:
    print("Tsunami Likely to Occur")
else:
    print("No Tsunami Expected")

print(f"Prediction Confidence: {max(prediction_proba[0]) * 100:.2f}%")