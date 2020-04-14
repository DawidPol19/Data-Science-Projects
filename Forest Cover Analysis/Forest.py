# -*- coding: utf-8 -*-
"""
Kaggle Forest Competition 

"""
#Importing Tools#
import pandas as pd
import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection
#Importing Data#
forest_data = pd.read_csv("train.csv")

#Object Creation#
y = forest_data.Cover_Type
features = ["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points","Wilderness_Area1","Wilderness_Area2","Wilderness_Area3","Wilderness_Area4",'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
X = forest_data[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Model Creation#
models = []
results = []
elist = []
elist_results = []
acc = []
cvres = []
msg = []
models.append(("KNN", KNeighborsClassifier(n_neighbors=1)))
models.append(("CART", DecisionTreeClassifier()))
models.append(("RFC", RandomForestClassifier(n_estimators=95, random_state=1)))
models.append(("XGB", XGBClassifier(n_estimators=95)))
models.append(("ETC", ExtraTreesClassifier(n_jobs=-1,n_estimators=100, random_state=1)))
#clf1 = KNeighborsClassifier(n_neighbors=2)
#clf2 = RandomForestClassifier(n_estimators=95, random_state=1)
#clf3 = XGBClassifier(n_estimators=95)
#models.append(("VTC", VotingClassifier(estimators=[("KNN", clf1), ("RFC", clf2), ("XGB", clf3)], voting="soft", weights=[1, 2, 1])))
#for name, model in models:
    #model.fit(X_train, y_train)
    #model_preds = model.predict(X_test)
    #model_mae = mean_absolute_error(model_preds, y_test)
    #model_accuracy = model.score(X_test, y_test)
    #results.append("{}: {:f}".format(name, model_mae))
    #acc.append("{}: {:f}".format(name, model_accuracy))
    #for i in range(len(list(y_test))):
        #if list(model_preds)[i] == list(y_test)[i]:
            #elist.append(True)
        #else:
            #elist.append(False)
    #elist_results.append("{}: {}/{}, {:f}".format(name, sum(elist), len(list(y_test)), sum(elist)/len(list(y_test))))
    #elist = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=7, random_state=1)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    cvres.append(cv_results)
    msg.append("{}: {:f} ({:f})".format(name, cv_results.mean(), cv_results.std()))

    
print("----------MAE-Validation----------")
for i in results:
    print(i)
print("-------------C.Accuracy-----------")
for i in acc:
    print(i)
print("-------------Accuracy-------------")
for i in elist_results:
    print(i)
print("----------------CV----------------")
for i in msg:
    print(i)
print("----------------------------------")

