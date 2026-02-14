import streamlit as st
import pandas as pd
import numpy as np

from model.preprocessing import encode_categorical, train_test_split
from model.metrics import *
from model.logistic_regression import LogisticRegressionScratch
from model.decision_tree import DecisionTreeScratch
from model.knn import KNNScratch
from model.naive_bayes import NaiveBayesScratch

st.title("Mushroom Classification ML App")

uploaded = st.file_uploader("Upload Test Dataset CSV", type="csv")

model_choice = st.selectbox("Select Model", [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes"
])

if uploaded:
    df = pd.read_csv(uploaded)
    df, _ = encode_categorical(df)

    X = df.drop("class", axis=1).values
    y = df["class"].values

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    if model_choice=="Logistic Regression":
        model = LogisticRegressionScratch()
    elif model_choice=="Decision Tree":
        model = DecisionTreeScratch()
    elif model_choice=="KNN":
        model = KNNScratch()
    else:
        model = NaiveBayesScratch()

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    tp, tn, fp, fn = confusion_matrix(y_test,y_pred)

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