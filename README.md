# Customer-Churn-Prediction
Objective: Predict whether a customer will churn based on historical data.

Tools: Python, Pandas, Scikit-learn, Matplotlib/Seaborn, Jupyter Notebook

Steps and Code:
Data Collection: Use a publicly available dataset (e.g., Telco Customer Churn dataset from Kaggle).

# Load the dataset
import pandas as pd
data = pd.read_csv('data/telco_customer_churn.csv')
Data Cleaning: Handle missing values, encode categorical variables, and normalize numerical features.

# Handle missing values
data.dropna(inplace=True)

# Encode categorical variables
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, columns=['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                                     'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                                     'DeviceProtection', 'TechSupport', 'StreamingTV', 
                                     'StreamingMovies', 'Contract', 'PaperlessBilling', 
                                     'PaymentMethod'], drop_first=True)

# Normalize numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(data[['tenure', 'MonthlyCharges', 'TotalCharges']])
Exploratory Data Analysis (EDA): Visualize data distributions, correlations, and identify key features impacting churn.
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of the target variable
sns.countplot(data['Churn'])
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.show()
Feature Engineering: Create new features if needed (e.g., tenure length categories).

# Create tenure length categories
data['tenure_category'] = pd.cut(data['tenure'], bins=[0, 12, 24, 48, 60, 72], labels=['0-1 Year', '1-2 Years', '2-4 Years', '4-5 Years', '5-6 Years'])
data = pd.get_dummies(data, columns=['tenure_category'], drop_first=True)
Modeling: Use different classification models (Logistic Regression, Random Forest, XGBoost) to predict churn.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Split the data
X = data.drop(['Churn'], axis=1)
y = data['Churn'].map({'Yes': 1, 'No': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Random Forest
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# XGBoost
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
Evaluation: Compare model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
# Evaluation metrics
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

# Logistic Regression
log_reg_metrics = evaluate_model(y_test, y_pred_log_reg)
print('Logistic Regression:', log_reg_metrics)

# Random Forest
rf_metrics = evaluate_model(y_test, y_pred_rf)
print('Random Forest:', rf_metrics)

# XGBoost
xgb_metrics = evaluate_model(y_test, y_pred_xgb)
print('XGBoost:', xgb_metrics)
