import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.Utility import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path  =  os.path.join('Delivery_Time_prediction/artifact','preprocessor.pkl')
            model_path  =  os.path.join('Delivery_Time_prediction/artifact','model.pkl')

            preprocessor  =  load_object(preprocessor_path)
            model  =  load_object(model_path)

            data_scaled  =  preprocessor.transform(features)

            pred  =  model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Delivery_person_Age:int,
                 Delivery_person_Ratings:float,
                 Weather_conditions:str,
                 Road_traffic_density:str,
                 Vehicle_condition:int,
                 Type_of_order:str,
                 Type_of_vehicle:str,
                 multiple_deliveries:int,
                 Festival:str,
                 City:str,
                 pickup_time:float,
                 Distance:float):
        
        

                 self.Delivery_person_Age   =   Delivery_person_Age
                 self.Delivery_person_Ratings =  Delivery_person_Ratings
                 self.Weather_conditions =  Weather_conditions
                 self.Road_traffic_density =  Road_traffic_density
                 self.Vehicle_condition =  Vehicle_condition
                 self.Type_of_order = Type_of_order
                 self.Type_of_vehicle  =  Type_of_vehicle
                 self.multiple_deliveries  =  multiple_deliveries
                 self.Festival  =  Festival
                 self.City  =  City
                 self.pickup_time  =  pickup_time
                 self.Distance  =  Distance
        
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict   =   {
            'Delivery_person_Age':[self.Delivery_person_Age],
            'Delivery_person_Ratings':[self.Delivery_person_Ratings],
            'Weather_conditions':[self.Weather_conditions],
            'Road_traffic_density' : [self.Road_traffic_density],
            'Vehicle_condition':  [self.Vehicle_condition] ,
            'Type_of_order' : [self.Type_of_order], 
            'Type_of_vehicle'  : [ self.Type_of_vehicle ] ,
            'multiple_deliveries' :  [  self.multiple_deliveries  ],
            'Festival'  :[ self.Festival ] ,
            'City': [  self.City],
            'pickup_time': [  self.pickup_time ] ,
            'Distance': [self.Distance  ]
        
            }
            df   =   pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
