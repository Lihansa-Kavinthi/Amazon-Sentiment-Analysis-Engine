import pandas as pd
import gzip
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --- 1. SETUP PATHS AND LOADING FUNCTIONS ---
# This is the exact path to your file
FILE_PATH = r"D:\1.Lihansa\Intern\Elevvo\Appliances.jsonl.gz"

def parse(path):
    """Generator to read the gzipped JSONL file line by line."""
    with gzip.open(path, 'rb') as g:
        for l in g:
            yield json.loads(l)

def get_df(path, num_samples=50000):
    """Converts JSON lines into a Pandas DataFrame with progress updates."""
    i = 0
    df_dict = {}
    print(f"📂 Attempting to open: {path}")
    
    try:
        for d in parse(path):
            df_dict[i] = d
            i += 1
            if i % 10000 == 0:
                print(f"📝 Progress: Loaded {i} reviews...")
            if i == num_samples:
                break
        
        print(f"✅ Successfully loaded {i} reviews.")
        return pd.DataFrame.from_dict(df_dict, orient='index')
    
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find the file at {path}")
        return None

# --- 2. EXECUTION START ---

# Check if file exists before starting
if os.path.exists(FILE_PATH):
    # Load data
    df = get_df(FILE_PATH)
    
    if df is not None:
        # Preprocessing: Keep only text and rating
        # In the 2023 dataset, the columns are usually 'text' and 'rating'
        df = df[['text', 'rating']].dropna(subset=['text'])

        # Filter out 3-star reviews and create binary sentiment
        df = df[df['rating'] != 3]
        df['sentiment'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)
        
        print(f"✅ Preprocessing complete. Final dataset size: {len(df)} samples.")

        # --- 3. VECTORIZATION ---
        print("🧮 Converting text to numbers (TF-IDF Vectorization)...")
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X = tfidf.fit_transform(df['text'])
        y = df['sentiment']

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- 4. MODEL TRAINING ---
        print("🚀 Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)

        print("🚀 Training Naive Bayes...")
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        nb_pred = nb.predict(X_test)

        # --- 5. RESULTS ---
        print("\n" + "="*30)
        print("FINAL RESULTS")
        print("="*30)
        print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.2f}")
        print(f"Naive Bayes Accuracy:         {accuracy_score(y_test, nb_pred):.2f}")
        print("\nDetailed Report (Logistic Regression):")
        print(classification_report(y_test, lr_pred))
else:
    print(f"❌ ERROR: The path '{FILE_PATH}' does not exist. Please check your D: drive.")


# Visualize the most frequent positive and negative words

import matplotlib.pyplot as plt
import numpy as np

# 1. Extract feature names and coefficients
feature_names = np.array(tfidf.get_feature_names_out())
# Logistic Regression coefficients (higher = positive, lower = negative)
coefs = lr.coef_.flatten()

# 2. Get indices of the top 10 negative and top 10 positive words
top_neg_indices = np.argsort(coefs)[:10]
top_pos_indices = np.argsort(coefs)[-10:]

# 3. Combine them for plotting
top_indices = np.concatenate([top_neg_indices, top_pos_indices])
top_words = feature_names[top_indices]
top_coefs = coefs[top_indices]

# 4. Create the plot
plt.figure(figsize=(12, 6))
colors = ['red' if c < 0 else 'green' for c in top_coefs]
plt.barh(top_words, top_coefs, color=colors)
plt.xlabel('Sentiment Strength (Coefficient Value)')
plt.title('Top 10 Negative vs Top 10 Positive Words in Appliances')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Bonus 2: Compared two different mathematical approaches (Linear vs. Probabilistic)
# Create a simple bar chart for comparison
models = ['Logistic Regression', 'Naive Bayes']
accuracies = [accuracy_score(y_test, lr_pred), accuracy_score(y_test, nb_pred)]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'orange'])
plt.ylim(0.8, 1.0) # Zoom in to see the difference clearly
plt.ylabel('Accuracy Score')
plt.title('Model Comparison: Logistic Regression vs Naive Bayes')

# Adding the exact numbers on top of the bars
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.005, f"{v:.2f}", ha='center', fontweight='bold')

plt.show()