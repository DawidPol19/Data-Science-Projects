# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 20:00:43 2019

@author: Lofu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

datas = pd.DataFrame({"Concentration": [1,3,6,9,11,15,18,25,28,31,36,40,43], #13 
                      "NaCl": [0.35, 0.67, 0.78, 0.93, 1.04, 1.10, 1.18, 1.23, 1.27, 1.32, 1.34, 1.37, 1.41]}, index=None)
datai = [1,2,3,4,5,6,7,8,9,10,11,12,13]
X1 = datas["Concentration"]
X2 = datas["NaCl"]
#help(LinearRegression)
plt.plot(datai, X1, "o")
plt.plot(datai, X2, "o")
plt.show()