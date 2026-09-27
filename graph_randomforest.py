import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("kidney_disease.csv")

# Basic cleaning (adjust if needed)
df = df.replace("?", pd.NA)
df = df.dropna()

# Convert target column if needed
df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})

# Set style
sns.set(style="whitegrid")

# ----------------------------------
# 1. Correlation Heatmap
# ----------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------------
# 2. CKD vs Non-CKD Count Plot
# ----------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='classification', data=df)
plt.title("CKD vs Non-CKD Count")
plt.xlabel("0 = No CKD, 1 = CKD")
plt.show()

# ----------------------------------
# 3. Age Distribution
# ----------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# ----------------------------------
# 4. Blood Pressure Boxplot
# ----------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x='classification', y='bp', data=df)
plt.title("Blood Pressure vs CKD")
plt.xlabel("0 = No CKD, 1 = CKD")
plt.show()

# ----------------------------------
# 5. Feature Importance (Random Forest)
# ----------------------------------
from sklearn.ensemble import RandomForestClassifier

X = df.drop('classification', axis=1)
y = df['classification']

model = RandomForestClassifier()
model.fit(X, y)

importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importance, y=features)
plt.title("Feature Importance")
plt.show()