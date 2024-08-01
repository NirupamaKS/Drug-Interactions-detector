import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
from utils import custom_tokenizer

# Load the dataset
data = pd.read_csv('interactions.csv')

# Create a pipeline for the model
model = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=custom_tokenizer)),
    ('classifier', MultinomialNB())
])

# Prepare the data
X = data[['drug1', 'drug2']].apply(lambda x: ','.join(x), axis=1)
y = data['interaction']

# Train the model
model.fit(X, y)

# Save the model to a file
with open('drug_interaction_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as 'drug_interaction_model.pkl'.")