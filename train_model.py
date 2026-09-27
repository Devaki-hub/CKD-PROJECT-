import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv("kidney_disease.csv")

# Replace missing values
df.replace('?', np.nan, inplace=True)

# Drop missing rows
df.dropna(inplace=True)

# Select features
features = ['age','bp','sg','al','su','hemo','pcv','wc','rc']
df = df[features + ['classification']]

# Convert target column
df['classification'] = df['classification'].map({'ckd':0, 'notckd':1})

# Convert all columns to numeric
for col in features:
    df[col] = pd.to_numeric(df[col])

# Split
X = df[features]
y = df['classification']

print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ PIPELINE (IMPORTANT PART)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(class_weight="balanced", random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Accuracy
print("Accuracy:", pipeline.score(X_test, y_test))

# Save pipeline (NOT model)
pickle.dump(pipeline, open("model.pkl", "wb"))

print("Model trained successfully with Pipeline ✅")