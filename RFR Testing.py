# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:39:30 2019

@author: Lofu
"""

#Importing Tools#
import pandas as pd
import numpy
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection

#Importing Data#
forest_data = pd.read_csv("train.csv")

y = forest_data.Cover_Type
features = ["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points","Wilderness_Area1","Wilderness_Area2","Wilderness_Area3","Wilderness_Area4",'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
#features = ["Elevation", "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points", "Horizontal_Distance_To_Hydrology", "Aspect", "Hillshade_9am", "Vertical_Distance_To_Hydrology", "Wilderness_Area4", "Hillshade_Noon", "Hillshade_3pm", "Slope", "Soil_Type10", "Soil_Type38", "Soil_Type3", "Wilderness_Area1", "Soil_Type39", "Soil_Type4", "Wilderness_Area3", "Soil_Type40", "Soil_Type2", "Soil_Type22", "Soil_Type30", "Soil_Type17", "Soil_Type13", "Soil_Type29", "Soil_Type23", "Soil_Type12", "Soil_Type32", "Soil_Type6", "Soil_Type11", "Soil_Type33", "Wilderness_Area2", "Soil_Type31", "Soil_Type24", "Soil_Type35", "Soil_Type1", "Soil_Type20", "Soil_Type5", "Soil_Type18", "Soil_Type16", "Soil_Type14", "Soil_Type37"]
X = forest_data[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

results = []
ires = []
yres = []
#clf1 = KNeighborsClassifier(n_neighbors=1)
#clf2 = RandomForestClassifier(n_estimators=92, random_state=1)
#clf3 = RandomForestClassifier(n_estimators=150, random_state=1) 
#model = VotingClassifier(estimators=[("KNN", clf1), ("RFC", clf2), ("RFC2", clf3)], voting="soft", weights=[1, 2, 2])
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True)
kfold = model_selection.KFold(n_splits=10, random_state=1)
cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
results.append(cv_results)
yres.append(cv_results.mean())
ires.append("{}: {:f} ({:f})".format(str("VTC"), cv_results.mean(), cv_results.std()))
print(ires)
model.fit(X, y)

