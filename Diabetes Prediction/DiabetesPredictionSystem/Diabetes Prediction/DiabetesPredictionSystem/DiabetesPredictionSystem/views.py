from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    df = pd.read_csv("D:\Projects\Diabetes Prediction System\diabetes.csv")
    # Outliers Handling
    q1 = df["BMI"].quantile(0.25)
    q3 = df["BMI"].quantile(0.75)
    iqr = q3 - q1

    # define the upper and lower bounds for outlier detection
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr

    # remove outliers from the dataset
    df2 = df[df['BMI'] < upper_bound]
    df2 = df2[df2['BMI'] > lower_bound]
    # Outliers Handling
    q1 = df["Age"].quantile(0.25)
    q3 = df["Age"].quantile(0.75)
    iqr = q3 - q1

    # define the upper and lower bounds for outlier detection
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr

    # remove outliers from the dataset
    df2 = df2[df2['Age'] < upper_bound]
    df2 = df2[df2['Age'] > lower_bound]

    X = df2.drop(["SkinThickness", "Outcome"], axis=1)
    y = df2["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    clr = LogisticRegression(max_iter=500)
    clr.fit(X_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = clr.predict([[val1, val2, val3, val5, val6, val7, val8]])

    res = ""
    if pred == 1:
        res = "Positive"
    else:
        res = "Negative"
    return render(request, 'predict.html', {'Result2': res})


def about(request):
    return render(request, 'about.html')
