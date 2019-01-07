#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:21:33 2018

@author: borek
"""
import pandas as np
from sklearn import svm
import numpy as np

# Visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

X = [[0, 0], [1, 1]]
y = [0, 1]

clf = svm.SVC(gamma='scale')
clf.fit(X, y)

clf.predict([[2., 2.]])
