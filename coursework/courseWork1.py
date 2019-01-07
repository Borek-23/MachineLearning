#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:37:05 2018

@author: borek
"""
import pandas as pd

df = pd.read_csv('/home/borek/Documents/MLandAI/coursework/dropped.csv')

#print(df.head(10))
#print(df.tail(10))


# Rename columns
df2 = df.rename(columns={'Overall effectiveness':'Overall Eff'})

print(df2.columns)

# This rename does not have to create new dataframe
df.rename(columns={'Overall effectiveness':'Overall Eff'}, inplace=True)
print(df.columns)

df[['']]