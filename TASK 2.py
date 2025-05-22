import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset: Customer reviews with labeled sentiments
data = {'review': ["I love this product!", 
                   "This is the worst purchase I've made.",
                   "Very good quality, I'm impressed.",
                   "Terrible experience, never buying again.",
                   "It's okay, but could be better."],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'neutral']}

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert sentiment labels to numerical values
sentiment_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
df['sentiment'] = df['sentiment'].map(sentiment_mapping)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Predict sentiment for new reviews
new_reviews = ["Absolutely fantastic!", "Horrible quality, not worth the money."]
new_X = vectorizer.transform(new_reviews)
new_predictions = model.predict(new_X)

for review, sentiment in zip(new_reviews, new_predictions):
    label = {1: 'positive', 0: 'negative', 2: 'neutral'}[sentiment]
    print(f"Review: '{review}' -> Sentiment: {label}")

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    words = text.split()  # Tokenization
    words = [word for word in words if word not in stopwords.words('english')]  # Stop-word removal
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return ' '.join(words)

# Example usage
sample_text = "I absolutely loved this product! Best purchase ever."
print(preprocess_text(sample_text))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
reviews = ["I love this product!",
           "Worst purchase ever.",
           "Excellent quality, highly recommend!",
           "Horrible experience, never again.",
           "It’s okay, but could be better."]
labels = [1, 0, 1, 0, 2]  # 1 = positive, 0 = negative, 2 = neutral

# Preprocess all reviews
processed_reviews = [preprocess_text(review) for review in reviews]

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_reviews)
y = labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

from sklearn.metrics import classification_report

print("Classification Report:\n", classification_report(y_test, y_pred))
      
