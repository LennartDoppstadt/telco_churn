# Telecom Churn Prediction App

This Streamlit web application allows users to explore telecom customer data, visualize churn behavior, and predict customer churn using a trained machine learning model. It also supports retraining the model and maintains history for predictions and model updates.

---

## Business Application

Telecom companies often suffer significant revenue loss due to customer churn. Traditionally, businesses might apply a **blanket retention strategy**, offering promotions to all customers regardless of their risk of leaving. This is costly and inefficient.

With this churn prediction model, businesses can adopt a **targeted strategy** by:
- Predicting which customers are most likely to churn.
- Allocating retention budgets more effectively.
- Improving customer retention and lifetime value with focused incentives.

**Comparison**:

| Strategy              | Description                            | Cost Impact      | Targeted? |
|-----------------------|----------------------------------------|------------------|-----------|
| Blanket Strategy      | Retain all customers equally           | High             | No        |
| Model-Driven Strategy | Focus retention on high-risk customers | Cost-efficient   | Yes       |

---

## Features

### 1. Dataset Overview
- View the raw dataset and basic statistical summaries.
- Perform feature transformation and data cleaning automatically.

### 2. Churn Prediction
- Input customer features (e.g., contract type, charges).
- Get prediction on whether the customer will churn.
- Probability score included.
- Prediction is logged in **prediction history**.

### 3. Data Visualization
- Explore relationships between churn and customer features.
- Use interactive plots to understand patterns.

### 4. Model Retraining
- Choose between Logistic Regression, Random Forest, XGBoost, and SVM.
- Tune hyperparameters using GridSearchCV.
- View updated performance metrics.
- Save and use retrained models instantly.
- Retraining events are logged in **retraining history**.

### 5. History Tracking
- Access past churn predictions and their inputs.
- View logs of previous retraining sessions and scores.

---

**Install dependencies**:
   ```bash
   pip install -r requirements.txt


