from operator import index#perform operations
from pydoc import describe# to describe the modules
from sqlite3 import Row# seperates rows and columns
from grpc import GenericRpcHandler
import matplotlib.pyplot as plt #to create the plots and graph
from sklearn.metrics import accuracy_score, mean_absolute_error
import streamlit as stt  
import pickle
import numpy as np
import pandas as pd #import dataset and make dataframe
from sklearn.model_selection import train_test_split  #training and testing split
from xgboost import XGBRegressor 
import seaborn as sns



# Giving name and title to web page
stt.write("""
# Calories Burned Prediction
Detect the Number of Calories Burned Using Machine Learning!
          """)

#Get the data
calories_data = pd.read_csv('calories_data.csv')

#set the subheader
stt.subheader("Data Information:")

#show data as a table
stt.dataframe(calories_data)

#show statistics on the data
stt.write(calories_data.describe())
#Changing Gender
stt.write(calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True))

#show data on the chart
stt.subheader('Gender wise Distribution Of Data in Dataset')
chart_data = pd.DataFrame(["Male","Female"], columns=["Gender"])
stt.bar_chart(chart_data)

stt.subheader('Age, Height, Weight, Duration, Heart_Rate and Body_Temperature Distribution Of DataSet')
stt.set_option('deprecation.showPyplotGlobalUse', False)
df = pd.DataFrame(calories_data[:200], columns = ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp'])
df.hist()
#plt.show()
stt.pyplot()

#splitting of data in x and y components
x=calories_data[["Gender","Age","Height","Weight","Duration","Heart_Rate","Body_Temp"]]
y=calories_data[["Calories"]]

#splitting into testing and training dataset
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=2)

#get user input
def get_user_input():
    gender = stt.sidebar.slider('Gender_Male_or_Female', 0, 1, 0)
    age	= stt.sidebar.slider('Yours_Age_in_Years', 15, 80, 30)
    height = stt.sidebar.slider('Yours_height_in_cm', 80, 180, 115)
    weight = stt.sidebar.slider('Yours_Weight_in_kg', 35, 95, 50)
    duration = stt.sidebar.slider('Duration_of_Exercise_in_minutes', 15, 40, 20)
    heart_rate = stt.sidebar.slider('Heart_Rate', 50, 122, 72)
    body_temp = stt.sidebar.slider('Body_Temp_in_Centigrade', 0.0, 60.0, 35.0)
    
    # store a dictionary under a variable
    user_data = { 'gender': gender,
             'age': age,
             'height': height,
             'weight': weight,
             'duration': duration,
             'heart_rate': heart_rate,
             'body_temp': body_temp
                }
    #transform features into dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features


#store user input to the variable
user_input = get_user_input()


#set subheader and then display the user input
stt.subheader('User Input: ')
stt.write(user_input)

#training the model
model = XGBRegressor()
model.fit(x_train,y_train)

#show model metric
stt.subheader('Model Absolute Mean Error: ')
stt.write(str(mean_absolute_error(y_test, model.predict(x_test))))

#store model prediction in the variable
test_data_prediction = model.predict(user_input)

#set a subheader and display the classification
stt.subheader('Your Predicted Burned Number of Calories are: ')
stt.write(test_data_prediction)
