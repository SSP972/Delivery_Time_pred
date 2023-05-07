import os
import sys
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
from math import radians, cos, sin, asin, acos, sqrt, pi
from geopy import distance
from geopy.geocoders import Nominatim
import osmnx as ox
import networkx as nx
from statistics import  mode
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score
        
        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)





def calculate_spherical_distance(lat1, lon1, lat2, lon2, r=6371):
    # Convert degrees to radians
    coordinates = abs(lat1), abs(lon1), abs(lat2), abs(lon2)
    # radians(c) is same as c*pi/180
    phi1, lambda1, phi2, lambda2 = [
        radians(c) for c in coordinates
    ]  
    
    # Apply the haversine formula
    a = (np.square(sin((phi2-phi1)/2)) + cos(phi1) * cos(phi2) * 
         np.square(sin((lambda2-lambda1)/2)))
    d = 2*r*asin(np.sqrt(a))
    return d


drop_list_pipe=['Restaurant_latitude',
        'Restaurant_longitude', 
        'Delivery_location_latitude',
        'Delivery_location_longitude',
        'Time_Orderd', 
        'Time_Order_picked', 
        'Order_Date',
        'ID',
        'Delivery_person_ID'

                ]
    # Column dropper
def dropper(df):
    df.drop(drop_list_pipe,inplace=True, axis=1)
    return df



#Distance converter
def distance_con_pipe(df):
    
        df['Distance']=[
            round(calculate_spherical_distance(*row), 2) 
            for row in df[['Restaurant_latitude', 'Restaurant_longitude', 
                        'Delivery_location_latitude', 
                        'Delivery_location_longitude']].values
                    ]
        return df


# Pickuptime function
def get_pickup_time(df):
    def time_to_minutes(x):
        if ((isinstance(x,str))and(":" in x)):
            return float(x.split(":")[0]) * 60 + (float(x.split(":")[1]))
        else:
            return np.nan


    df['Time_Order_picked']=df['Time_Order_picked'].apply(time_to_minutes)
    df['Time_Orderd']=df['Time_Orderd'].apply(time_to_minutes)
    df['pickup_time']=df['Time_Order_picked']-df['Time_Orderd']
    return df

#input data given to the model by user 
def input_data_collector(input_data_path:str,final_new_data):
    try:
        collected_data=pd.read_csv(input_data_path)
        
        df = pd.concat([collected_data, final_new_data], ignore_index=True)
        df.to_csv(input_data_path,index=False)
    except FileNotFoundError :
        final_new_data.to_csv(input_data_path,index=False)
        logging.info('data collection file is created successfully')