# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:34:58 2018

@author: s15106137
"""


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

dropped = normaliseAllColumns(dropped)
dropped.to_csv('processedData.csv')
print("done")