import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from model.preprocessing import encode_categorical
from model.metrics import *



# Load models from pickle files with caching
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Logistic Regression": "models/Logistic_Regression.pkl",
        "Decision Tree": "models/Decision_Tree.pkl",
        "KNN": "models/KNN.pkl",
        "Naive Bayes": "models/Naive_Bayes.pkl",
        "Random Forest (Ensemble)": "models/Random_Forest_Ensemble.pkl",
        "XGBoost (Ensemble)": "models/XGBoost_Ensemble.pkl"
    }
    
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
        else:
            st.error(f"Model file not found: {model_path}")
            st.info("Please run `python train_models.py` to train and save models first.")
    
    return models


st.title("Mushroom Classification Application")

# Load pre-trained models
models = load_models()

if not models:
    st.error("No models loaded. Please run `python train_models.py` first.")
    st.stop()

# Download test data section
st.sidebar.subheader("📥 Download Test Data")
test_data_df = pd.read_csv("data/mushrooms-test-data.csv")
csv_data = test_data_df.to_csv(index=False)
st.sidebar.download_button(
    label="Download mushrooms-test-data.csv",
    data=csv_data,
    file_name="mushrooms-test-data.csv",
    mime="text/csv"
)

uploaded = st.file_uploader("Upload Test Dataset CSV", type="csv")

model_choice = st.selectbox("Select Model", list(models.keys()))

if uploaded:
    df = pd.read_csv(uploaded)
    df, _ = encode_categorical(df)

    X_test = df.drop("class", axis=1).values
    y_test = df["class"].values

    # Get the selected pre-trained model
    model = models[model_choice]
    
    # Make predictions
    y_pred = model.predict(X_test)

    tp, tn, fp, fn = confusion_matrix(y_test,y_pred)
    
    # Display confusion matrix
    cm_data = np.array([[tp, fp], [fn, tn]])
    cm_df = pd.DataFrame(cm_data, 
                         index=['Actual Positive', 'Actual Negative'],
                         columns=['Predicted Positive', 'Predicted Negative']).astype(str)
    st.subheader("Confusion Matrix")
    st.dataframe(cm_df)

    acc = accuracy(tp,tn,fp,fn)
    prec = precision(tp,fp)
    rec = recall(tp,fn)
    f1 = f1_score(prec,rec)
    mcc_score = mcc(tp,tn,fp,fn)

    # Display metrics as table
    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "MCC"],
        "Score": [f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{mcc_score:.4f}"]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.subheader("Performance Metrics")
    st.dataframe(metrics_df, hide_index=True)
    