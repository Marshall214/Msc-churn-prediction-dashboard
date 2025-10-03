# Predicting and Reducing Customer Churn in Local E-Commerce Platforms

### An Ensemble Machine Learning and Natural Language Processing Approach

This repository contains the implementation and supporting files for my MSc Project: **“Predicting and Reducing Customer Churn in Local E-Commerce Platforms: An Ensemble Machine Learning and Natural Language Processing Approach”**. The project integrates **structured transaction data** and **unstructured customer reviews**, applying **ensemble machine learning models** and **NLP techniques** to predict churn and provide actionable insights. A **Streamlit dashboard** is also included to demonstrate the practical application of the model in a decision-support context.

---

## Repository Structure

```
├── Msc_Project.ipynb           # Main research notebook (exploration, preprocessing, modeling, evaluation)
├── gomart_churn.csv            # Structured churn dataset
├── gomart_reviews.csv          # Unstructured review dataset
├── model.pkl                   # Trained XGBoost churn prediction model
├── streamlit_app.py            # Streamlit dashboard source code
├── requirements.txt            # Dependencies for reproducibility
├── README.md                   # Project documentation (this file)
└── thesis.pdf                  # Full MSc thesis (Predicting and Reducing Customer Churn...)
```

---

## ⚙️ Features

* **Data Preprocessing**: Cleaning, encoding, scaling, and feature engineering (RFM metrics, sentiment scores).
* **NLP Integration**: Sentiment analysis of customer reviews with polarity features feeding into churn models.
* **Imbalance Handling**: Applied SMOTE, class weighting, and threshold tuning to address skewed churn classes.
* **Modeling**: Compared Logistic Regression, Decision Trees, Random Forest, and **XGBoost** (best performer).
* **Evaluation**: Metrics include Accuracy, Precision, Recall, F1/F2 scores, AUC-ROC, and PR-AUC.
* **Interpretability**: Explainable AI with **SHAP** to identify key churn drivers.
* **Deployment**: Interactive **Streamlit dashboard** showcasing predictions, feature importance, and insights.

---

## Getting Started

### 1️Clone the Repository

```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
```

### Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # On Mac/Linux
venv\Scripts\activate       # On Windows
```

### 3️Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️Run the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

The app will launch in your default browser at:
 `http://localhost:8501/`

---

## Streamlit Dashboard Overview

The dashboard provides:

* **Customer Data Upload**: Upload CSVs for predictions.
* **Predictions**: Churn likelihood score for each customer.
* **Feature Importance (SHAP)**: Visual explanations of churn drivers.
* **Data Insights**: Key statistics and churn distribution.
* **Actionable Insights**: Highlighting strategies for retention.

---

## Model

* **Algorithm**: XGBoost (Gradient Boosting)
* **Serialized Model**: Stored as `model.pkl`
* **Explainability**: SHAP summary plots and force plots

You can load and use the trained model in Python as follows:

```python
import joblib

# Load model
model = joblib.load("model.pkl")

# Predict
pred = model.predict(X_test)
```

---

## Results

* **Best Model**: XGBoost
* **Performance**:

  * Accuracy: ~0.995
  * F1 Score (Churn Class): ~0.99
  * AUC-ROC: ~0.99
* **Top Churn Drivers**:

  * Low Recency (time since last purchase)
  * Negative Sentiment in reviews
  * Low frequency of transactions
  * Failed/delayed deliveries

---

## Future Work

* Real-time deployment with APIs and live transaction streams.
* Deeper NLP integration (topic modeling, contextual embeddings like BERT).
* Cross-domain datasets for improved generalizability.
* Integration with CRM systems for automated retention campaigns.

---

## Acknowledgements

This project was completed as part of my MSc in Data Science at **Pan-Atlantic University (School of Science and Technology)**. I extend gratitude to my supervisor, faculty, and colleagues for their support.

---

## License

This repository is for **academic purposes**. Commercial use requires permission.
