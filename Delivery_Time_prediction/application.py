from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
import os
from src.logger import logging
import pandas as pd
from src.Utility import input_data_collector




application=Flask(__name__)

app=application

import os 
# print(os.chdir(r'Delivery_Time_prediction'),

# os.getcwd())

@app.route('/')
def home_page():
    
    return render_template(['index.html'])

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
                        Delivery_person_Age = request.form.get('Delivery_person_Age'),
                        Delivery_person_Ratings = request.form.get('Delivery_person_Ratings'),
                        Weather_conditions =  request.form.get('Weather_conditions'),
                        Road_traffic_density =  request.form.get('Road_traffic_density'),
                        Vehicle_condition = request.form.get('Vehicle_condition'),
                        Type_of_order =  request.form.get('Type_of_order'),
                        Type_of_vehicle =  request.form.get('Type_of_vehicle'),
                        multiple_deliveries = request.form.get('multiple_deliveries'),
                        Festival =  request.form.get('Festival'),
                        City =  request.form.get('City'),
                        pickup_time = request.form.get('pickup_time'),
                        Distance = request.form.get('Distance')
            
                        )

        input_data_path=os.path.join('Delivery_Time_prediction/artifact','inputdata.csv')
        final_new_data=data.get_data_as_dataframe()
        logging.info(f'{final_new_data.head(1)}')
        
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)                    
        final_new_data['Time_taken (min)']=results                      # Collection of data from client
        input_data_collector(input_data_path, final_new_data) 
        
        return render_template('results.html',final_result=results)
@app.route('/train',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return 

if __name__=="__main__":
    app.run(host='0.0.0.0',port=8000,debug=True)
    
