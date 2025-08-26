# Customer Churn Prediction for a Telecom Company 📊

## 🚀 Live Demo
**You can try the interactive web app here:** [**Streamlit App Link**](https://customer-churn-prediction-gxtodszh8pjk58rvawbfzg.streamlit.app)


## 1. Business Problem
This project aims to proactively identify customers who are most likely to churn (cancel their subscription). By predicting churn, the company can develop targeted retention strategies to reduce revenue loss and improve customer loyalty.


## 2. Key Findings & Visualizations
Analysis of the data revealed several key factors that influence churn. The most significant finding is that the customer's contract type is the primary driver of churn.

<img width="1200" height="800" alt="feature_importance" src="https://github.com/user-attachments/assets/f5fc7ebc-72ab-4e0b-ad5f-e91e2915d723" />


## 3. Model Performance
An XGBoost Classifier was trained to predict customer churn. The model's performance on the unseen test data is summarized below:

**Classification Report:**
          precision    recall  f1-score   support
       0       0.87      0.78      0.82      1035
       1       0.53      0.67      0.59       374
accuracy                           0.75      1409
macro avg       0.70      0.73      0.71      1409
weighted avg       0.78      0.75      0.76      1409

* **AUC-ROC Score**: 0.8187
* The model successfully identifies **67% of the customers who actually churned** (Recall score), making it a valuable tool for targeted marketing campaigns.


## 4. Actionable Insights & Recommendations
Based on the model's findings, here are three actionable recommendations:

* **Insight 1:** Customers with **month-to-month contracts** are the most likely to churn.
    * **Recommendation:** Develop a marketing campaign to incentivize these customers to switch to one-year or two-year contracts, potentially offering a small discount or a free service upgrade.

* **Insight 2:** **Low customer tenure** is a major predictor of churn.
    * **Recommendation:** Implement an enhanced customer onboarding program for the first 90 days to ensure new customers feel valued and understand their service benefits.

* **Insight 3:** Customers with **Fiber optic internet** have a higher churn rate.
    * **Recommendation:** The company should investigate potential issues with its Fiber optic service, such as reliability or pricing, as it is clearly linked to higher churn.


## 5. How to Run this Project
1. Clone the repository.
2. Install the necessary libraries: `pip install pandas scikit-learn xgboost matplotlib seaborn`
3. Run the script: `python churn_prediction.py`
