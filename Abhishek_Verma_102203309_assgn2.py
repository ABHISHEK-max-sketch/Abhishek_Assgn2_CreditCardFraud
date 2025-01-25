#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from the provided URL
data_url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
data = pd.read_csv(data_url)

# Split into features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Combine the balanced dataset
balanced_data = pd.concat([pd.DataFrame(X_balanced), pd.DataFrame(y_balanced, columns=['Class'])], axis=1)

# Define sampling techniques
def simple_random_sampling(df):
    return df.sample(frac=1, random_state=42)

def stratified_sampling(df):
    return df.groupby('Class', group_keys=False).apply(lambda x: x.sample(frac=0.8, random_state=42))

def systematic_sampling(df):
    return df.iloc[::5, :]

def cluster_sampling(df):
    return df.sample(frac=0.5, random_state=42)

def bootstrap_sampling(df):
    return df.sample(n=len(df), replace=True, random_state=42)

# List of sampling methods with their names for heatmap labels
sampling_methods = [
    ("Simple Random Sampling", simple_random_sampling),
    ("Stratified Sampling", stratified_sampling),
    ("Systematic Sampling", systematic_sampling),
    ("Cluster Sampling", cluster_sampling),
    ("Bootstrap Sampling", bootstrap_sampling)
]

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Store results
results = []

# Iterate over sampling methods and models
for sampling_name, sampling_method in sampling_methods:
    sampled_data = sampling_method(balanced_data)
    X_sample = sampled_data.drop('Class', axis=1)
    y_sample = sampled_data['Class']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model in models.items():
        # Change scoring to 'accuracy' instead of 'f1'
        cv_scores = cross_val_score(model, X_sample, y_sample, cv=skf, scoring='accuracy')
        accuracy_score = np.mean(cv_scores)
        results.append({
            "Sampling Technique": sampling_name,
            "Model": model_name,
            "Accuracy Score": accuracy_score
        })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv("sampling_results_accuracy.csv", index=False)

# Pivot the results for heatmap
results_pivot = results_df.pivot(index="Model", columns="Sampling Technique", values="Accuracy Score")

# Print the Accuracy Score results
print("Accuracy Score Results Table:")
print(results_pivot)

# Plot the heatmap for Accuracy Scores
plt.figure(figsize=(10, 6))
sns.heatmap(results_pivot, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Accuracy Score of Models with Different Sampling Techniques")
plt.ylabel("Model")
plt.xlabel("Sampling Technique")
plt.show()

# Plot the comparison of average Accuracy scores across models
average_accuracy_scores = results_df.groupby("Model")["Accuracy Score"].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=average_accuracy_scores.index, y=average_accuracy_scores.values, palette="viridis")
plt.title("Average Accuracy Score Comparison Across Models")
plt.xlabel("Models")
plt.ylabel("Average Accuracy Score")
plt.xticks(rotation=45)
plt.show()


# In[ ]:




