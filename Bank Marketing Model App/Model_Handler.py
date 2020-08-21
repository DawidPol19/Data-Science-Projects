# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:17:03 2020
@author: dawid
"""

#Silencing Future Warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

#Basic Utilities
import seaborn as sns
sns.set()
import joblib

#Model Utilities
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

#Model Handle Object
class Model_Handler():
    def __init__(self, path):
        self.path = path
        self.data = None
        self.model = None
        
    def preprocess_data(self):
        obj_features = ['job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_week','poutcome', 'y']
    
        #Data Encoding for Classification
        data_encode = self.data.copy()
        label_dict = {}  
        for i in obj_features:
            label_dict[i] = LabelEncoder().fit(data_encode[i])
            data_encode[i] = label_dict[i].transform(data_encode[i])
    
        #Selecting Features For Modelling
        y = data_encode["y"]
        train_features = ["age","job","marital","education","default","housing",
                         "loan","contact","month","day_of_week","duration",
                         "campaign","pdays","previous","poutcome","emp.var.rate",
                         "cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]
        X = data_encode[train_features]
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
    
        return X_train, X_valid, y_train, y_valid
    
    def train_model(self, X_train, X_valid, y_train, y_valid):
        model = LGBMClassifier(n_estimators=70, n_jobs=-1)
        model.fit(X_train, y_train)
        validation_accuracy = model.score(X_valid, y_valid)
        return model, validation_accuracy
    
    def save_model(self, model):
        joblib.dump(model, 'bank_marketing_classifier.mdl')
        
    def load_model(self):
        if 'bank_marketing_classifier.mdl' in os.listdir(self.path):
            self.model = joblib.load('bank_marketing_classifier.mdl')
        else:
            print('Cannot Load Model, File Missing')
    
    def model_predict(self, predict_data):
        obj_features = ['job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_week','poutcome']
        label_dict = {}  
        for i in obj_features:
            label_dict[i] = LabelEncoder().fit(self.data[i])
            predict_data[i] = label_dict[i].transform(predict_data[i])
            
        pred_features = ["age","job","marital","education","default","housing",
                         "loan","contact","month","day_of_week","duration",
                         "campaign","pdays","previous","poutcome","emp.var.rate",
                         "cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]
        preds = self.model.predict(predict_data[pred_features])
        self.data = None
        return preds



