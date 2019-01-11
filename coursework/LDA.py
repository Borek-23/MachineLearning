# -*- coding: utf-8 -*-
"""
Linear Discriminant Analysis
"""
VERBOSE = False
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def printScore(score_array, label,modelName):
    avg = np.mean(score_array)
    std = np.std(score_array)
    print("{:s}. {:s}. Mean: {:f} - Standard Deviation: {:f}".format(modelName,label,avg,std))
    
def visualise(modelName, X, y):
    import matplotlib.pyplot as plt
    # This is providing styles for plotting
    from matplotlib import style
    # Style specifier
    style.use('ggplot')
    
    #plt.legend(loc=4)
    #recall_score.plot()
    #precision_score.plot()
    X['0'].plot()
    y.plot()
    plt.xlabel('Schools')
    plt.ylabel('Score')
    plt.show()
 
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
        plotModel(model, X.iloc[train], y.iloc[train])
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
    if(modelName == "NB"):
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        return model.fit(X_train, y_train)
    if(modelName == "Support Vector Machine"):
        from sklearn import svm
        clf = svm.SVC(C=0.5, kernel='rbf', decision_function_shape='ovr')
        return clf.fit(X_train, y_train)
    
#Code Used to plot graph may not work with all models
def plotModel(model, X_train, y_train):
    import matplotlib.pyplot as plt
    
    # Sets the names of all the results
    target_names = ['Outstanding','Good','Requires Improvement','Inadequate']
    # Sets the colours for all the results
    colours = ['Green', 'lime', 'darkorange', 'red']

    X_r = model.transform(X_train)
    
    x = np.array(X_r[:,0])
    y = np.array(X_r[:,1])
     
    
    plt.figure()
    # For each Y value it will plot the scatter graph with a seperate colour
    for colour, i, target_name in zip(colours, [1, 2, 3, 4], target_names):
        # X_r[y_train == i] checks all the value with the targets and returns all values that would return target i
        # the 0 or 1 is used to return either the x or y value to plot on the axis
        train = y_train == i
        train = np.array(train['0'])
        
        plt.scatter(x[train], y[train], alpha=.8, color=colour,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    # Sets the plot title
    plt.title('Linear Discriminant Analysis of Ofsted Dataset')
    
    # Displays the plot
    plt.show()

#Main method tying together the whole flow and defines file/column names
def ofsted():
    import pandas as pd
    X = pd.read_csv('X.csv')
    X = X.drop("Unnamed: 0",axis=1)
    y = pd.read_csv('y.csv')
    y = y.drop("Unnamed: 0",axis=1)
    
    foldData("LDA", X, y)

    
#Use preprocessing.py first to create processedData.csv
ofsted()

