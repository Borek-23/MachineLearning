# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:34:58 2018

@author: s15106137
"""

VERBOSE = True


import pandas as pd
grouped = pd.read_csv('csv/england_cfrgrouped.csv')
ofsted = pd.read_csv('csv/england_ofsted-schools.csv')
joined = ofsted.set_index('URN').join(grouped.set_index('URN'),how="inner")
joined.to_csv('joined.csv')
print(joined.columns)

def normaliseColumn(data, columnName):
    colMin = data[columnName].min()
    colMax = data[columnName].max()
    
    def norm(item):
        return (item - colMin)/ (colMax - colMin)
    data[columnName] = data[columnName].apply(norm)
    return data

def normaliseAllColumns(data):
    exluded = ["URN", "Overall effectiveness"]
    for column in data:
        if (column not in exluded):
            data = normaliseColumn(data, column)
    return data     
def removeComma(x):
    if (type(x) is str):
        try:
            return float(x.split()[0].replace(',', ''))
        except:
            return x
    return x

def removeCommas(dataframe):
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].apply(removeComma)
    return dataframe

def removeValue(dataframe, value):
    def remove(x):
        if (x == value):
            return 0
        return x
    
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].apply(remove)
    return dataframe

def convertColumnToFloat(dataframe, columnname):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(dataframe[columnname].astype(str))
    dataframe[columnname] = le.transform(dataframe[columnname].astype(str))
    return dataframe
    
joined = joined.drop("Unnamed: 33",axis=1)
joined = joined.drop("School name (as shown on Performance tables)",axis=1)
joined = joined.drop("Old school name (if different)",axis=1)
joined = joined.drop("LA name",axis=1)
joined = joined.drop("LA code",axis=1)
joined = joined.drop("Estab code",axis=1)
joined = joined.drop("School DfE number",axis=1)
joined = joined.drop("Phase of education",axis=1)
joined = joined.drop("Phase for median group",axis=1)
joined = joined.drop("TOTAL INCOME (£ per pupil)",axis=1)
joined = joined.drop("TOTAL EXPENDITURE (£ per pupil)",axis=1)

dropped = joined
print(dropped.columns)
dropped = convertColumnToFloat(dropped, "Establishment type")
dropped = convertColumnToFloat(dropped, "Region")
dropped = convertColumnToFloat(dropped, "London / Non-London")
dropped = convertColumnToFloat(dropped, "FSM band")
dropped = removeCommas(dropped)
dropped = removeValue(dropped, "SUPP")
dropped = removeValue(dropped, "..")
dropped = removeValue(dropped, "")
dropped = dropped.dropna(0, 'any')

dropped = normaliseAllColumns(dropped)#
print(type(dropped))
dropped.to_csv('processedData.csv')
print("done")


def smoteIt(X, y):
    from imblearn.over_sampling import SMOTE
    if(VERBOSE):
        print("Before SMOTE: {0} values".format(len(y.index)))
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)
    if(VERBOSE):
        print("After SMOTE: {0} values".format(len(y)))
    return X, y


def splitData(data, target_field):
    y = data.loc[:,[target_field]]
    X = data.drop(target_field, axis=1)
    return X, y


X, y = splitData(dropped, "Overall effectiveness")
smoteX, smotey = smoteIt(X, y)
Xframe = pd.DataFrame(smoteX)
print(Xframe)
yframe = pd.DataFrame(smotey)
Xframe.to_csv('X.csv')
yframe.to_csv('y.csv')