import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv(r"D:\Desktop\2ND SEM PROJECTS\CROP RECOMMENDATION POJECT\Crop_recommendation.csv")  # Ensure the CSV is in your project folder

# Features and labels
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(model, 'crop_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("✅ Model and label encoder saved successfully!")

# train_model.py

def get_response(user_input):
    user_input = user_input.lower()

    responses = {
        "hello": "Hi there! How can I assist you with farming today?",
        "hi": "Hello! Need help with crops or irrigation?",
        "what crop should i grow": "You can use the prediction feature on the homepage to get a crop suggestion.",
        "which fertilizer is good": "It depends on your soil type and crop. Common choices are Urea, DAP, and NPK.",
        "how to protect crops from pests": "Use integrated pest management, crop rotation, and proper pesticide use.",
        "what is smart irrigation": "Smart irrigation uses sensors and automation to optimize water use in farming.",
        "thank you": "You're welcome! Happy farming!",
        "bye": "Goodbye! Feel free to come back anytime.",
    }

    for key in responses:
        if key in user_input:
            return responses[key]

    return "I'm sorry, I didn't understand that. Please ask something related to farming."


# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
# Load dataset
df = pd.read_csv(r"D:\Desktop\2ND SEM PROJECTS\CROP RECOMMENDATION POJECT\Crop_recommendation.csv")  # Ensure the CSV is in your project folder
# Features and labels
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Save model and label encoder
joblib.dump(model, 'crop_model.pkl')