# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 20:19:24 2020
@author: dawid
"""
#Run this python script before starting the app

#Data and SQL Utilities
import pandas as pd
import sqlite3
from sqlite3 import Error

#Name of the Database
db_file = 'bank_marketing.db'

#Open Initial CSV File
init_data = pd.read_csv('Data\\bank-additional-full.csv', delimiter=';')

#Create Table in Sqlite3
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

#Insert Data into the Table        
def insert_init_data(db_file, init_data):
    conn = create_connection(db_file)
    init_data.to_sql('bank_marketing_main', conn)
    return None

#Run The Initialization
if __name__ == '__main__':
    insert_init_data(db_file, init_data)


    

        

    
    


