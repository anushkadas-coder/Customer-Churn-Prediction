# ==============================================================================
# Section 1: Import Necessary Libraries
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

print("Libraries imported successfully.")


# ==============================================================================
# Section 2: Load and Initially Inspect the Data
# ==============================================================================
# Load the dataset
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully.")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found.")
    print("Please download the dataset from Kaggle and place it in the same directory as this script.")
    exit()

# Initial data inspection
print("\nDataset Info:")
df.info()

# The 'TotalCharges' column is object type, needs to be converted to numeric.
# Some values are empty strings ' ' which will cause an error. We handle this.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check for missing values after coercion
print("\nMissing values in each column:")
print(df.isnull().sum())

# We will handle the missing 'TotalCharges' by imputing with the median.
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
print("\nMissing 'TotalCharges' filled with median.")

# Drop the customerID column as it's not a useful feature for prediction
df.drop('customerID', axis=1, inplace=True)
print("\n'customerID' column dropped.")

# Convert target variable 'Churn' to binary (0/1)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
print("\nTarget variable 'Churn' converted to 0/1.")


# ==============================================================================
# Section 3: Exploratory Data Analysis (EDA)
# ==============================================================================
print("\nStarting Exploratory Data Analysis... Plots will be saved as PNG files.")

# Set plot style
sns.set_style("whitegrid")

# 1. Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution (0 = No Churn, 1 = Churn)')
plt.savefig('eda_churn_distribution.png')
plt.close()
print("Saved 'eda_churn_distribution.png'")

# 2. Churn by Contract Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type')
plt.savefig('eda_churn_by_contract.png')
plt.close()
print("Saved 'eda_churn_by_contract.png'")

# 3. Churn by Internet Service
plt.figure(figsize=(10, 6))
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title('Churn by Internet Service Type')
plt.savefig('eda_churn_by_internet_service.png')
plt.close()
print("Saved 'eda_churn_by_internet_service.png'")

# 4. Tenure distribution for Churn vs. No Churn
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30, kde=True)
plt.title('Tenure Distribution by Churn Status')
plt.savefig('eda_tenure_distribution.png')
plt.close()
print("Saved 'eda_tenure_distribution.png'")

print("\nEDA finished.")


# ==============================================================================
# Section 4: Data Preprocessing
# ==============================================================================
print("\nStarting data preprocessing...")

# Define features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

print(f"\nNumerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# Create preprocessing pipelines for numerical and categorical data
# Numerical features will be scaled.
# Categorical features will be one-hot encoded.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


# ==============================================================================
# Section 5: Model Training
# ==============================================================================
print("\nSplitting data and training the model...")

# Split the data into training and testing sets (80/20 split)
# We use stratify=y to ensure the proportion of churn is the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate scale_pos_weight for handling class imbalance in XGBoost
# It's the ratio of negative class to positive class
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"\nCalculated 'scale_pos_weight' for imbalance: {scale_pos_weight:.2f}")

# Create the final machine learning pipeline with preprocessing and the model
# The model is an XGBoost Classifier, which is powerful for this type of problem
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight, # Handle class imbalance
        random_state=42
    ))
])

# Train the model
model.fit(X_train, y_train)
print("\nModel training complete.")


# ==============================================================================
# Section 6: Model Evaluation
# ==============================================================================
print("\nEvaluating the model on the test set...")

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Print Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('evaluation_confusion_matrix.png')
plt.close()
print("Saved 'evaluation_confusion_matrix.png'")

# Print AUC-ROC Score
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC Score: {auc_score:.4f}")


# ==============================================================================
# Section 7: Feature Importance
# ==============================================================================
print("\nGenerating feature importance plot...")

# Get feature names after one-hot encoding
feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numerical_features, feature_names])

# Get feature importances from the trained XGBoost model
importances = model.named_steps['classifier'].feature_importances_

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importances
}).sort_values('importance', ascending=False).head(15) # Top 15 features

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('Top 15 Most Important Features for Churn Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
print("Saved 'feature_importance.png'")

print("\n--- SCRIPT FINISHED ---")
print("Check the folder for saved plots: 'eda_*.png', 'evaluation_*.png', and 'feature_importance.png'")