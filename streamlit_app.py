from model.xgboost import XGBoostClassifier
import streamlit as st
import pandas as pd
import numpy as np

from model.preprocessing import encode_categorical, train_test_split
from model.metrics import *
from model.logistic_regression import LogisticRegressionScratch
from model.decision_tree import DecisionTreeScratch
from model.knn import KNNScratch
from model.naive_bayes import NaiveBayesScratch
from model.random_forest import RandomForestScratch
# from model.xgboost import XGBoostScratch



st.title("Mushroom Classification Application")

# Load and train models with training data
training_df = pd.read_csv("data/mushrooms-training-data.csv")
training_df, _ = encode_categorical(training_df)

X_train = training_df.drop("class", axis=1).values
y_train = training_df["class"].values

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

model_choice = st.selectbox("Select Model", [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest (Ensemble)",
    "XGBoost (Ensemble)"
])

if uploaded:
    df = pd.read_csv(uploaded)
    df, _ = encode_categorical(df)

    X_test = df.drop("class", axis=1).values
    y_test = df["class"].values

    if model_choice=="Logistic Regression":
        model = LogisticRegressionScratch()
    elif model_choice=="Decision Tree":
        model = DecisionTreeScratch()
    elif model_choice=="KNN":
        model = KNNScratch()
    elif model_choice=="Naive Bayes":
        model = NaiveBayesScratch()
    elif model_choice=="Random Forest (Ensemble)":
        model = RandomForestScratch(random_state=42)
    else:
        model = XGBoostClassifier(random_state=25)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    tp, tn, fp, fn = confusion_matrix(y_test,y_pred)
    
    # Display confusion matrix
    cm_data = np.array([[tp, fp], [fn, tn]])
    cm_df = pd.DataFrame(cm_data, 
                         index=['Actual Positive', 'Actual Negative'],
                         columns=['Predicted Positive', 'Predicted Negative'])
    st.subheader("Confusion Matrix")
    st.dataframe(cm_df)

    acc = accuracy(tp,tn,fp,fn)
    prec = precision(tp,fp)
    rec = recall(tp,fn)
    f1 = f1_score(prec,rec)
    mcc_score = mcc(tp,tn,fp,fn)

    st.write("Accuracy:", acc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)
    st.write("F1:", f1)
    st.write("MCC:", mcc_score)
    