Amazon Product Sentiment Analysis 🚀
📖 Project Overview
Developed during my internship at Elevvo, this project addresses the challenge of extracting meaningful consumer sentiment from large-scale, unstructured product review data. Using the UCSD Amazon Appliances dataset, I built a pipeline that classifies reviews as Positive or Negative with 92% accuracy.

🔍 Technical Scope
Big Data Handling: Processed 50,000+ records from compressed .jsonl.gz files using memory-efficient Python generators.

Linguistic Engineering: Transformed raw text into a high-dimensional mathematical matrix using TF-IDF Vectorization (5,000 features).

Predictive Modeling: Compared Logistic Regression and Multinomial Naive Bayes to find the most robust classifier for short-form consumer text.

💻 Implementation Highlights
The project is structured into 5 logical stages:

1. Data Streaming
Instead of unzipping the file manually, the code streams data directly from the .gz container to save disk space and RAM.

2. Preprocessing
We cleaned the data by removing 3-star (neutral) reviews to create a clear binary classification target and handled missing text values.

3. Model Training
```python
# Logistic Regression was the top performer
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
```
4. Interpretation (The "Why")
We used model coefficients to identify the "trigger words" for sentiment.

Top Positive: efficient, easy, perfect, highly

Top Negative: disappointed, return, waste, defective

5. Deployment Logic
The model is serialized into .pkl files for instant loading and inference.

📊 Performance Results
Model,Accuracy,Precision (Neg),Recall (Pos)
Logistic Regression,92%,0.84,0.99
Naive Bayes,90%,0.81,0.97

🛠️ How to Run
Clone the repo and ensure you have the Appliances.jsonl.gz file in the directory.

Install Dependencies:
pip install pandas scikit-learn matplotlib joblib

Run the Notebook: Open Sentiment_Analysis.ipynb to see the live visualizations and training steps.

Try Custom Input: Use the predict_sentiment() function inside the script to test your own sentences!
