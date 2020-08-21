# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:15:39 2020
@author: dawid
"""

#Import Basic Utilities
import sqlite3
from sqlite3 import Error
import pandas as pd

#SQL Handle Object
class SQL_Handler():
    def __init__(self, db_file, db_table, sql_script):
        self.db_file = db_file
        self.db_table = db_table
        self.sql_script = sql_script
        
    def create_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            return conn
        except Error as e:
            print(e) 
        return conn
    
    def select_database(self):
        if self.sql_script == None:
            conn = self.create_connection()
            view_frame = pd.read_sql_query('SELECT * FROM {}'.format(self.db_table), conn)
            return view_frame
        else:
            view_frame = pd.read_sql_query(self.sql_script, conn)
            return view_frame
    
    def custom_sql(self, sql_script):
        conn = self.create_connection()
        cur = conn.cursor()
        cur.execute(self.sql_script)
        conn.commit()
        self.sql_script = None
        
        
