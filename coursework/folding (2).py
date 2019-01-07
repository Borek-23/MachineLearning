# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:57:10 2018

@author: s15106137
"""
VERBOSE = False
import warnings
warnings.filterwarnings('ignore')

def printScore(score_array, label,modelName):
    import numpy as np
    avg = np.mean(score_array)
    std = np.std(score_array)
    print("{:s}. {:s}. Mean: {:f} - Standard Deviation: {:f}".format(modelName,label,avg,std))
 

#Evaluates model and gives f1-score. Generic to all algorithms
def scoreModel(model, X_test, y_test):
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    y_pred = model.predict(X_test)
    from sklearn.metrics import classification_report
    if(VERBOSE):
        print(classification_report(y_test,y_pred))
    return [f1_score(y_test, y_pred,average='weighted'), precision_score(y_test, y_pred,average='weighted'), recall_score(y_test, y_pred,average='weighted')]
   
def foldData(modelName, X,y):
    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(10, True, 1)
    f1_score_array = []
    precision_score_array = []
    recall_score_array = []
    for train, test in kfold.split(X, y):
        model = trainModel(modelName, X.iloc[train], y.iloc[train])
        scores = scoreModel(model, X.iloc[test], y.iloc[test])
        f1_score_array.append(scores[0])
        precision_score_array.append(scores[1])
        recall_score_array.append(scores[2])
    printScore(f1_score_array, "F1 Score",modelName)
    printScore(precision_score_array, "Precision Score",modelName)
    printScore(recall_score_array, "Recall Score",modelName)
    
#Method for training a specific model. Returns the model MODIFY FOR EACH ALGORITHM
def trainModel(modelName, X_train, y_train):
    if(modelName == "Decision Tree"):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0)
        return model.fit(X_train, y_train)
    if(modelName == "Neural Network"):  
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(24,24,24))
        return mlp.fit(X_train,y_train)
    if(modelName == "LDA"):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis(n_components=2)
        return model.fit(X_train, y_train)
    if(modelName == "Support Vector Machine"):
        from sklearn import svm
        clf = svm.SVC(C=0.5, kernel='rbf', decision_function_shape='ovr')
        return clf.fit(X_train, y_train)

#Main method tying together the whole flow and defines file/column names
def ofsted():
    import pandas as pd
    X = pd.read_csv('X.csv')
    X = X.drop("Unnamed: 0",axis=1)
    y = pd.read_csv('y.csv')
    y = y.drop("Unnamed: 0",axis=1)
    #foldData("Decision Tree", X, y)
    #foldData("Neural Network", X, y)
    #foldData("LDA", X, y)  
    foldData("Support Vector Machine", X, y)
    
#Use preprocessing.py first to create processedData.csv
ofsted()

