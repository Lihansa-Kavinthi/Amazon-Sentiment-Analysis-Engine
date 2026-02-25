# Amazon Product Sentiment Analysis 🚀

## 📖 Project Overview
Developed during my internship at **Elevvo**, this project addresses the challenge of extracting meaningful consumer sentiment from large-scale, unstructured product review data. Using the UCSD Amazon Appliances dataset, I built a pipeline that classifies reviews as **Positive** or **Negative** with **92% accuracy**.

## 📊 Dataset
Due to file size limitations, the raw dataset (`Appliances.jsonl.gz`) is hosted externally.  
👉 **[Download the Amazon Product Dataset here](https://drive.google.com/file/d/1oU1MgjeyxcgYq_wKU83HblwYbo645pBf/view?usp=sharing)** *Note: Please ensure the file is placed in the project root directory before running the scripts.*

## 🔍 Technical Scope
* **Big Data Handling**: Processed 50,000+ records from compressed `.jsonl.gz` files using memory-efficient Python generators.
* **Linguistic Engineering**: Transformed raw text into a high-dimensional mathematical matrix using **TF-IDF Vectorization** (5,000 features).
* **Predictive Modeling**: Compared **Logistic Regression** and **Multinomial Naive Bayes** to find the most robust classifier for short-form consumer text.

## 💻 Implementation Highlights
The project is structured into 5 logical stages:

1. **Data Streaming**: Instead of unzipping the file manually, the code streams data directly from the `.gz` container to save disk space and RAM.
2. **Preprocessing**: We cleaned the data by removing 3-star (neutral) reviews to create a clear binary classification target and handled missing text values.
3. **Model Training**: 
   ```python
   # Logistic Regression was the top performer
   lr = LogisticRegression(max_iter=1000)
   lr.fit(X_train, y_train)
   
### 4. Interpretation (The "Why")
We analyzed the model coefficients to identify the "trigger words" that carry the most weight in determining sentiment.
* **Top Positive Words**: *efficient, easy, perfect, highly, great*
* **Top Negative Words**: *disappointed, return, waste, defective, terrible*

### 5. Deployment Logic
The model and vectorizer are serialized into `.pkl` files using `joblib`. This allows for **instant loading and inference** in production environments without the need to retrain on the 50,000-row dataset every time.

---

## 📊 Performance Results

| Model | Accuracy | Precision (Neg) | Recall (Pos) |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **92%** | 0.84 | 0.99 |
| **Naive Bayes** | **90%** | 0.81 | 0.97 |



---

## 🛠️ How to Run

### 1. Prerequisites
Clone the repository and ensure the dataset `Appliances.jsonl.gz` is located in the project root directory.

### 2. Install Dependencies
Run the following command in your terminal to install the necessary libraries:
```bash
pip install pandas scikit-learn matplotlib joblib
```

### 3. Execution

* **Interactive Mode**: Open `Sentiment_Analysis.ipynb` in VS Code or Jupyter Notebook. This is recommended for viewing data visualizations and step-by-step logic.

* **Direct Script**: To run the full pipeline (loading, training, and evaluation) in one go, execute the following in your terminal:
  ```bash
  python Amazon.py
  predict_sentiment("This product exceeded my expectations!")

---
*Transforming raw data into digital insights, one line of code at a time. 🚀*



