import pickle
import pandas as pd
import numpy as np
from model.preprocessing import encode_categorical
from model.logistic_regression import LogisticRegressionScratch
from model.decision_tree import DecisionTreeScratch
from model.knn import KNNScratch
from model.naive_bayes import NaiveBayesScratch
from model.random_forest import RandomForestScratch
from model.xgboost import XGBoostClassifierScratch

# Load and preprocess training data
print("Loading training data...")
training_df = pd.read_csv("data/mushrooms-training-data.csv")
training_df, _ = encode_categorical(training_df)

X_train = training_df.drop("class", axis=1).values
y_train = training_df["class"].values

# Define models
models = {
    "Logistic Regression": LogisticRegressionScratch(),
    "Decision Tree": DecisionTreeScratch(),
    "KNN": KNNScratch(),
    "Naive Bayes": NaiveBayesScratch(),
    "Random Forest (Ensemble)": RandomForestScratch(random_state=42),
    "XGBoost (Ensemble)": XGBoostClassifierScratch(random_state=25)
}

# Train and save models
print("Training models...")
for model_name, model in models.items():
    print(f"  Training {model_name}...")
    model.fit(X_train, y_train)
    
    # Save to pickle file with standardized naming
    pkl_filename_map = {
        "Logistic Regression": "models/Logistic_Regression.pkl",
        "Decision Tree": "models/Decision_Tree.pkl",
        "KNN": "models/KNN.pkl",
        "Naive Bayes": "models/Naive_Bayes.pkl",
        "Random Forest (Ensemble)": "models/Random_Forest_Ensemble.pkl",
        "XGBoost (Ensemble)": "models/XGBoost_Ensemble.pkl"
    }
    
    pkl_filename = pkl_filename_map[model_name]
    with open(pkl_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved to {pkl_filename}")

print("All models trained and saved!")
