# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:37:37 2018

@author: s15106137
"""

#Splits data into X and Y, as well as test and training. Generic to all algorithms
def splitData(data, target_field):
    from sklearn.cross_validation import train_test_split
    train, test = train_test_split(data, test_size=0.3, random_state=0)
    y_train = train.loc[:,[target_field]]
    X_train = train.drop(target_field, axis=1)
    y_test = test.loc[:,[target_field]]
    X_test = test.drop(target_field, axis=1)
    return X_train, y_train, X_test, y_test

#Method for training a specific model. Returns the model MODIFY FOR EACH ALGORITHM
def trainModel(X_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0)
    return tree.fit(X_train, y_train)

#Evaluates model and gives f1-score. Generic to all algorithms
def scoreModel(model, X_test, y_test):
    predictions = model.predict(X_test)
    from sklearn.metrics import classification_report
    print(classification_report(y_test,predictions))
    
#Main method tying together the whole flow and defines file/column names
def ofsted():
    import pandas as pd
    data = pd.read_csv('processedData.csv')
    X_train, y_train, X_test, y_test = splitData(data, "Overall effectiveness") 
    model = trainModel(X_train, y_train)
    scoreModel(model, X_test, y_test)
    
    
#Use preprocessing.py first to create processedData.csv
ofsted()