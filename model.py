# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and prepare data
df = pd.read_csv('drug_interactions.csv')
X = df[['drug_1', 'drug_2']]
y = df['interaction']

# Encode the drugs (you'll need to handle this appropriately)
X_encoded = pd.get_dummies(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'interaction_model.pkl')
