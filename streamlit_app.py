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


@st.cache_resource
def load_mappings():
    mappings_path = "models/mappings.pkl"
    if os.path.exists(mappings_path):
        with open(mappings_path, 'rb') as f:
            return pickle.load(f)
    st.error(f"Mappings file not found: {mappings_path}")
    st.info("Please run `python train_models.py` to train and save models first.")
    return None


st.title("Mushroom Classification Application")

# Load pre-trained models
models = load_models()
mappings = load_mappings()

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
    # Encode test data using mappings generated from training data
    if mappings is not None:
        df, _ = encode_categorical(df, mappings=mappings)
    else:
        df, _ = encode_categorical(df)

    X_test = df.drop("class", axis=1).values
    y_test = df["class"].values

    # Get the selected pre-trained model
    model = models[model_choice]
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Compute predicted probabilities for AOC if possible
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
        # If predict_proba returns 2D, use positive class column
        if y_pred_proba.ndim == 2:
            # Find positive class code from mappings
            if mappings is not None and 'class' in mappings:
                class_map = mappings['class']
                # Find which code is positive (edible/poisonous): assume max code is positive
                pos_label = max(class_map.values())
                neg_label = min(class_map.values())
                # st.write(f"[DEBUG] Class mapping: {class_map}, pos_label={pos_label}, neg_label={neg_label}")
                pos_col = list(class_map.values()).index(pos_label)
            else:
                pos_label = 1
                neg_label = 0
                pos_col = 1
            y_pred_proba_pos = y_pred_proba[:, pos_col]
        else:
            y_pred_proba_pos = y_pred_proba
            pos_label = 1
            neg_label = 0
        # Compute AOC
        from model.metrics import aoc_score
        aoc = aoc_score(y_test, y_pred_proba_pos, pos_label=pos_label, debug=True)
        # st.write(f"[DEBUG] y_pred_proba_pos[:10]: {y_pred_proba_pos[:10]}")
        # st.write(f"[DEBUG] y_test[:10]: {y_test[:10]}")
        # st.write(f"[DEBUG] AOC (AUC) score: {aoc:.4f}")
    else:
        aoc = None
        # st.write("[DEBUG] Model does not support probability prediction. AOC not computed.")

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
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "MCC", "AOC (AUC)"],
        "Score": [f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{mcc_score:.4f}", f"{aoc:.4f}" if aoc is not None else "N/A"]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.subheader("Performance Metrics")
    st.dataframe(metrics_df, hide_index=True)
    