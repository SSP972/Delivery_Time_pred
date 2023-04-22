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
from src.Utility import save_object,calculate_spherical_distance

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('Delivery_price_prediction/artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):
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

            logging.info('Pipeline Initiated')
            
            
            # Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
            
                ('imputer',SimpleImputer(strategy='mean')),
                ('scaler',StandardScaler())
                    
                    ] 
                                )
            # Categorical Pipeline
            cat_pipeline_ODE =Pipeline(
                steps=[
                
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[Weather_conditions_ODE,Road_traffic_density_ODE,Type_of_vehicle_ODE,Festival_ODE])),
                ('scaler',StandardScaler())
                
                ]
                                )
            cat_pipeline_OHE=Pipeline(
                steps=[
                    ('onehotencoder',OneHotEncoder(categories=[OHE_Cat_City,OHE_Cat_type_orders]))
                ]
            )
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline_ODE',cat_pipeline_ODE,['Weather_conditions', 'Road_traffic_density','Type_of_vehicle', 'Festival']),
            ('cat_pipeline_OHE',cat_pipeline_OHE,['City','Type_of_order'])
            ])

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info('Transforming Longitudes and latitudes into distance')
            #Distance Transformation

            
            distancepreprocessing = Pipeline([
                    ('distance', FunctionTransformer(lambda x: x.assign(Distance=[round(calculate_spherical_distance(*row), 2) 
                    for row in x[['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']].values]))),
                    ('drop_cols', FunctionTransformer(lambda x: x.drop(['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude'], axis=1)))
                ])

            train_df= distancepreprocessing.fit_transform(train_df)
            test_df=distancepreprocessing.fit_transform(test_df)
            logging.info('Distance is transformed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()
            
            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name,'ID']
            
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