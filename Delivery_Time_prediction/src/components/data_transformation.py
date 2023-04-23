import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline,FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder

from src.exception import CustomException
from src.logger import logging
import os
from src.Utility import save_object,calculate_spherical_distance,dropper,month_spliter,get_pickup_time,distance_con_pipe

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('Delivery_Time_prediction/artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            
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


            logging.info('Data transformation Pipeline Initiated')


            #Distance Transformation pipeline
            distance_pipeline = Pipeline(steps=[
                ('Distance_converter',FunctionTransformer(distance_con_pipe))
                ])
    
            # Pipeline for date column
            date_pipeline = Pipeline(steps=[
                ('extract_month', FunctionTransformer(month_spliter))
                                    ])

            # Pipeline for time columns
            time_pipeline = Pipeline(steps=[
                ('time_conversion', FunctionTransformer(get_pickup_time))
                                    ])


            drop_non_essential=Pipeline(steps=[
                ('drop_cols', FunctionTransformer(dropper)),
            
            ])
            # # Combine pipelines
            full_pipeline = Pipeline([
                ('distance_preprocessing', distance_pipeline),
                ('time_pipeline', time_pipeline),
                ('date_pipeline', date_pipeline),    
                ('Drop_non_essential', drop_non_essential)   

            ])

            logging.info('Data transfrmation Pipeline Completed')
            return full_pipeline
        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        





    def get_data_transformation_preprocessing_object(self):
        try:
            logging.info('Data Transformation initiated')
            
         
            #Categories for ordinal encoding
            Weather_conditions_ODE=['Sunny','Cloudy','Windy','Fog', 'Stormy', 'Sandstorms' ]
            Road_traffic_density_ODE=['Low', 'Medium', 'High', 'Jam']
            Type_of_vehicle_ODE=['bicycle', 'electric_scooter', 'scooter', 'motorcycle']
            Festival_ODE=['No','Yes']

            #Categories for One Hot encoding
            OHE_Cat_City=['Metropolitian', 'Urban', 'Semi-Urban']
            OHE_Cat_type_orders=['Snack', 'Meal', 'Drinks', 'Buffet']

            #Column Transformation
            num_CT=['Delivery_person_Age', 'Delivery_person_Ratings','Vehicle_condition','pickup_time','Distance','Order_Month']
            ordinal_CT=['Weather_conditions', 'Road_traffic_density','Type_of_vehicle', 'Festival']
            OHE_CT=['City','Type_of_order']


            logging.info('Pipeline Initiated')
            

            #Pipeline for frequent values handling 
            frequncy_of_delivery=Pipeline(
                steps = [
                ('imputer',SimpleImputer(strategy='most_frequent'))
                
                        ])

            # Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
            
                ('imputer',SimpleImputer(strategy='mean')),
                ('scaler',StandardScaler(with_mean=False))
                    
                    ] 
                                )
        # Categorical Pipeline
            cat_pipeline_ODE =Pipeline(
                steps=[
                
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[Weather_conditions_ODE,Road_traffic_density_ODE,Type_of_vehicle_ODE,Festival_ODE])),
                ('scaler',StandardScaler(with_mean=False))
                
                ]
                                )
            cat_pipeline_OHE=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder',OneHotEncoder(categories=[OHE_Cat_City,OHE_Cat_type_orders])),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            # combined pre processing pipeline
            preprocessor=ColumnTransformer([

            ('num_pipeline',num_pipeline,num_CT),
            ('cat_pipeline_ODE',cat_pipeline_ODE,ordinal_CT),
            ('cat_pipeline_OHE',cat_pipeline_OHE,OHE_CT),
            ('Frequency_match',frequncy_of_delivery,[ 'multiple_deliveries'])
            
            ])

            
            logging.info('Pipeline Completed')
            return preprocessor
        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)



    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_preprocessing_object()
            data_transformation = self.get_data_transformation_object()
            
            
            train_df=data_transformation.fit_transform(train_df)            # converting the data set into required data
            test_df=data_transformation.fit_transform(test_df)              # converting the data set into required data

            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name]
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)


