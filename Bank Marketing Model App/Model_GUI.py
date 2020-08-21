# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 19:02:36 2020
@author: dawid
"""

#########Paths#################################################################
#Put the path of where the folder is located for the file chooser
path = 'C:\\Program Files'
preds_path = 'Predictions'
graph_path = 'Graphs'

##########Import###############################################################
#Import Kivy Utilities
import kivy
print(kivy.version)

from kivy.app import App
from kivy.uix.label import Label
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import BooleanProperty
from kivy.uix.recycleview import RecycleView
from kivy.uix.recyclegridlayout import RecycleGridLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior

#Data, Graph and Basic Utilities
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Custom Objects
from Model_Handler import Model_Handler
from SQL_Handler import SQL_Handler

#############KV Script Loading#################################################
Builder.load_file('Model_GUI.kv')

########Model Handle###########################################################
model_handle = Model_Handler(path)
###############################################################################

########SQL Handle#############################################################
sql_handle = SQL_Handler('bank_marketing.db', 'bank_marketing_main', None)
###############################################################################

########View Screen############################################################
class CustomLabel(Label):
    pass

class ViewScreen(Screen):
    pass

class RV(RecycleView):
    def __init__(self, **kwargs):
        super(RV, self).__init__(**kwargs)
        self.rangeid_lower = 0
        self.rangeid_upper = 51
        data_csv = sql_handle.select_database()
        table_data = []
        for y in data_csv.columns:
            table_data.append({'text':str(y), 'height':20})
            
        for i in range(self.rangeid_lower, self.rangeid_upper):
            for x in data_csv.iloc[i]:
                table_data.append({'text':str(x), 'height':20})
        self.data = table_data
        del data_csv, table_data
        
    def change_range(self, range_input):
        rng = range_input.split('-')
        if len(rng) == 1:
            self.rangeid_lower = 0
            self.rangeid_upper = len(self.data) if int(rng[0]) > len(self.data) else int(rng[0])+1
        elif len(rng) == 2:
            self.rangeid_lower = 0 if int(rng[0]) < 0 else int(rng[0])
            self.rangeid_upper = len(self.data) if int(rng[1]) > len(self.data) else int(rng[1])+1
        data_csv = sql_handle.select_database()
        table_data = []
        for y in data_csv.columns:
            table_data.append({'text':str(y), 'height':20})
            
        for i in range(self.rangeid_lower, self.rangeid_upper):
            for x in data_csv.iloc[i]:
                table_data.append({'text':str(x), 'height':20})
        self.data = table_data
        del data_csv, table_data

#######Modify Screen###########################################################
class ModifyScreen(Screen):
    pass

class ModifyScreenAddCsv(Screen):
    def __init__(self, **kwargs):
        super(ModifyScreenAddCsv, self).__init__(**kwargs)
        self.file_chooser.rootpath = path
    
    def press(self):
        new_label = 'Adding to Database...'
        self.progress_label.text = new_label
        
    def released(self, filename):
        try:
            data = pd.read_csv(filename)
            conn = sql_handle.create_connection()
            data.to_sql(sql_handle.db_table, conn, if_exists='append')
            new_label = 'CSV File Added Successfully!'
            self.progress_label.text = new_label
        except:
            new_label = 'CSV File Adding Failed!'
            self.progress_label.text = new_label
            
    def clear_label(self):
        new_label = ''
        self.progress_label.text = new_label
        
class ModifyScreenCustomSQL(Screen):
    def press(self):
        new_label = 'Running SQL Command...'
        self.progress_label.text = new_label
    
    def run_custom_sql(self, text_input):
        try:
            sql_handle.sql_script = text_input
            sql_handle.custom_sql(text_input)
            new_label = 'SQL Command Completed'
            self.progress_label.text = new_label
        except:
            new_label = 'SQL Command Failed'
            self.progress_label.text = new_label

class ModifyScreenRetrain(Screen):
    def press(self):
        new_label = 'Retraining Model...'
        self.progress_label.text = new_label
    
    def retrain_model(self):
        try:
            model_handle.data = sql_handle.select_database()
            X_train, X_valid, y_train, y_valid = model_handle.preprocess_data()
            os.replace('bank_marketing_classifier.mdl', 'old_bank_marketing_classifier.mdl')
            model, validation_acc = model_handle.train_model(X_train, X_valid, y_train, y_valid)
            model_handle.save_model(model)
            model_handle.data = None
            del X_train, X_valid, y_train, y_valid, model, 
            new_label = 'Model Trained Successfully! \nModel Accuracy: \n{:.3f}'.format(validation_acc)
            self.progress_label.text = new_label
        except:
            new_label = 'Model Retraining Failed'
            self.progress_label.text = new_label
            

#######Predict Screen##########################################################
class PredictScreen(Screen):
    pass

#######Predict CSV Screen######################################################
class PredictCSVScreen(Screen):
    def __init__(self, **kwargs):
        super(PredictCSVScreen, self).__init__(**kwargs)
        self.file_chooser.rootpath = path
    
    def pressed(self):
        new_label = 'Running Prediction...'
        self.progress_label.text = new_label
    
    def released(self, filename):
        if '.csv' in filename:
            model_handle.data = sql_handle.select_database()
            self.pred_data = pd.read_csv(filename)
            model_handle.load_model()
            preds = model_handle.model_predict(self.pred_data)
            self.pred_data['Predicted Y'] = preds
            pred_file_name = (filename.split("\\")[::-1])[0]
            pred_file_name = pred_file_name.replace('.csv', '_Predicted_Results.csv')
            self.pred_data.to_csv(pred_file_name)
            os.replace(pred_file_name, preds_path + '\\' + pred_file_name)
            new_label = 'Prediction Finished\nExported to: {}'.format(pred_file_name)
            self.progress_label.text = new_label
        else:
            self.progress_label.text = 'Wrong File Selected, Try Again'
    
    def clear_label(self):
        new_label = ''
        self.progress_label.text = new_label
        pass

#######Predict Excel Screen####################################################
class PredictExcelScreen(Screen):
    def __init__(self, **kwargs):
        super(PredictExcelScreen, self).__init__(**kwargs)
        self.file_chooser.rootpath = path
    
    def pressed(self):
        new_label = 'Running Prediction...'
        self.progress_label.text = new_label
    
    def released(self, filename):
        try:
            if '.xlsx' in filename:
                model_handle.data = sql_handle.select_database()
                self.pred_data = pd.read_excel(filename)
                model_handle.load_model()
                preds = model_handle.model_predict(self.pred_data)
                self.pred_data['Predicted Y'] = preds
                pred_file_name = (filename.split("\\")[::-1])[0]
                pred_file_name = pred_file_name.replace('.xlsx', '_Predicted_Results.xlsx')
                self.pred_data.to_excel(pred_file_name)
                os.replace(pred_file_name, preds_path + '\\' + pred_file_name)
                new_label = 'Prediction Finished\nExported to: {}'.format(pred_file_name)
                self.progress_label.text = new_label
            else:
                self.progress_label.text = 'Wrong File Selected, Try Again'
        except:
            new_label = 'Prediction Failed! Try Again'
            self.progress_label.text = new_label
    
    def clear_label(self):
        new_label = ''
        self.progress_label.text = new_label
        pass
    
##########Graph Screen#########################################################
#Object to handle index selection between Graph and Graph Display
class Index_Measure():
    def __init__(self):
        self.index_x = 'None'
        self.index_y = 'None'
Index_Measure = Index_Measure()

#Recycle Layout with Selectable Labels
class SelectableRecycleGridLayout(FocusBehavior, LayoutSelectionBehavior,
                                 RecycleGridLayout):
    pass

#Selectable Label for X Axis
class SelectableLabel1(RecycleDataViewBehavior, Label):
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def refresh_view_attrs(self, rv, index, data):
        self.index = index
        return super(SelectableLabel1, self).refresh_view_attrs(
            rv, index, data)

    def on_touch_down(self, touch):
        if super(SelectableLabel1, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        self.selected = is_selected
        if is_selected:
            Index_Measure.index_x = index


#Selectable Label for Y Axis            
class SelectableLabel2(RecycleDataViewBehavior, Label):
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def refresh_view_attrs(self, rv, index, data):
        self.index = index
        return super(SelectableLabel2, self).refresh_view_attrs(
            rv, index, data)

    def on_touch_down(self, touch):
        if super(SelectableLabel2, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        self.selected = is_selected
        if is_selected:
            Index_Measure.index_y = index

#X Axis Scrollable Recycle View
class GraphRecycleView_1(RecycleView):
    def __init__(self, **kwargs):
        super(GraphRecycleView_1, self).__init__(**kwargs)
        self.table_columns = sql_handle.select_database().columns
        self.data = [{'text':'None', 'height':33}] + [{'text':str(x), 'height':33} for x in self.table_columns]
    
#Y Axis Scrollable Recycle View
class GraphRecycleView_2(RecycleView):
    def __init__(self, **kwargs):
        super(GraphRecycleView_2, self).__init__(**kwargs)
        self.table_columns = sql_handle.select_database().columns
        self.data = [{'text':'None', 'height':33}] + [{'text':str(x), 'height':33} for x in self.table_columns]

#Final GraphScreen
class GraphScreen(Screen):
    def reset_indices(self):
        Index_Measure.index_x = 'None'
        Index_Measure.index_y = 'None'
    
    def create_graph(self):
        graph = 'None'
        data = sql_handle.select_database()
        graph_columns = ['None'] + list(data.columns)
        x, y = 'None', 'None'
        if Index_Measure.index_x != 'None':
            x = graph_columns[Index_Measure.index_x]
        if Index_Measure.index_y != 'None':
            y = graph_columns[Index_Measure.index_y]
            
        if x == 'None':
            if y == 'None':
                graph = 'None'
                
            elif data[y].dtype == 'int64' or data[y].dtype == 'float64':
                plt.figure(figsize=(10,9))
                graph = sns.distplot(data[y], 30)
                plt.savefig('temp_graph.png')
                plt.cla()
                del data, graph, graph_columns
                return None
            
            elif data[y].dtype == 'object':
                plt.figure(figsize=(10,9))
                graph = sns.countplot(x=None, y=y, data=data)
                plt.savefig('temp_graph.png')
                plt.cla()
                del data, graph, graph_columns
                return None
                
        elif data[x].dtype == 'int64' or data[x].dtype == 'float64':
            if y == 'None':
                plt.figure(figsize=(11,9))
                graph = sns.distplot(data[x], 30)
                plt.savefig('temp_graph.png')
                plt.cla()
                del data, graph, graph_columns
                return None
            
            elif data[y].dtype == 'int64' or data[y].dtype == 'float64':
                plt.figure(figsize=(10,9))
                graph = sns.scatterplot(x, y, data=data)
                plt.savefig('temp_graph.png')
                plt.cla()
                del data, graph, graph_columns
                return None
            
            elif data[y].dtype == 'object':
                plt.figure(figsize=(10,9))
                graph = sns.catplot(x, y, data=data, kind='box', height=8)
            
        elif data[x].dtype == 'object':
            if y == 'None':
                plt.figure(figsize=(10,9))
                graph = sns.countplot(x, data=data)
                plt.savefig('temp_graph.png')
                plt.cla()
                del data, graph, graph_columns
                return None
            
            elif data[y].dtype == 'int64' or data[y].dtype == 'float64':
                plt.figure(figsize=(10,9))
                graph = sns.catplot(x, y, data=data, kind='box', height=8)
                
            elif data[y].dtype == 'object':
                data_dic = {}
                data[y] = data[y].astype('category')
                for i in data[x].unique():
                    sub_data = data[data[x] == i]
                    data_dic[i] = list(sub_data[y].value_counts(dropna=False))
                new_frame = pd.DataFrame(data=data_dic, index=data[y].unique())
                plt.figure(figsize=(10,9))
                graph = sns.heatmap(data=new_frame, annot=True, fmt='d')
                plt.savefig('temp_graph.png')
                plt.cla()
                del data, graph, graph_columns, data_dic, new_frame
                return None
        
        if graph != 'None':
            graph.savefig('temp_graph.png')
        else:
            graph = sns.scatterplot([1,2,3], [1,2,3])
            plt.savefig('temp_graph.png')
              
        plt.cla()
        del data, graph, graph_columns
        return None

########Graph Display##########################################################
class GraphScreenDisplay(Screen):
    def del_temp(self):
        self.graph_display_label.text = 'File Name'
        self.graph_display_text.text = 'Saved_Figure.png'
        if 'temp_graph.png' in os.listdir(path):
            os.remove('temp_graph.png')       
    
    def save_figure(self, input_text):
        try:
            os.replace('temp_graph.png', graph_path + '\\' + input_text)
            self.graph_display_label.text = 'Figure Saved as {}'.format(input_text)
        except:
            self.graph_display_label.text = 'Error in Saving'
        

######Screen Manager###########################################################
sm = ScreenManager()
sm.add_widget(ViewScreen(name='view'))
sm.add_widget(ModifyScreen(name='modify'))
sm.add_widget(ModifyScreenAddCsv(name='modifycsv'))
sm.add_widget(ModifyScreenCustomSQL(name='modifysql'))
sm.add_widget(ModifyScreenRetrain(name='modifyretrain'))
sm.add_widget(PredictScreen(name='predict'))
sm.add_widget(PredictCSVScreen(name='predictcsv'))
sm.add_widget(PredictExcelScreen(name='predictexcel'))
sm.add_widget(GraphScreen(name='graph'))
sm.add_widget(GraphScreenDisplay(name='graphdisplay'))

######Running App##############################################################
class MarketingMLApp(App):
    def build(self):
        return sm

######Main Function############################################################
if __name__ == '__main__':
    MarketingMLApp().run()
    #Temporary Image Cleanup
    if 'temp_graph.png' in os.listdir(path):
            os.remove('temp_graph.png')

