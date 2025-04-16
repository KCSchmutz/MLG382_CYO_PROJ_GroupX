import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from dash import Dash, html, dcc, Input, Output
from functools import lru_cache

@lru_cache(maxsize=1)
def get_DLmodel():
    print("Loading deep learning model...")
    return load_model(r'artifacts/deep_learning_model.h5') 

with open('../artifacts/features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('../artifacts/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('../artifacts/numerical_columns.pkl', 'rb') as f:
    numerical_cols = pickle.load(f)

with open(r'../artifacts/RandomForestRegressor_model.pkl', 'rb') as f:
    randomforest_model = pickle.load(f)

with open(r'../artifacts/AdaBoostRegressor_model.pkl', 'rb') as f:
    regression_model = pickle.load(f)

with open(r'../artifacts/XGBRegressor_model.pkl', 'rb') as f:
    xgboost_model = pickle.load(f)


# with open(r'../artifacts/GradientBoostingRegressor_model.pkl', 'rb') as f:
#     gradboost_model = pickle.load(f)


app = Dash(__name__)
server = app.server

# Sample input layout
app.layout = html.Div([
    html.H1("Student Grade Predictor"),
    html.P("Enter values below to predict the grade class"),
    dcc.Input(id='age', type='number', placeholder='Age', value=18),
    dcc.Dropdown(
            id='gender', 
            options=[{'label': 'Male', 'value': 0}, {'label': 'Female', 'value': 1}],
            placeholder="Select Gender", value=0),
    dcc.Input(id='study_time', type='number', placeholder='Study Time Weekly', value=15),
    dcc.Input(id='absences', type='number', placeholder='Absences', value=5),
    dcc.Input(id='gpa', type='number', placeholder='GPA', value=2),
    dcc.Dropdown(
        id='ethnicity',
        options=[{'label': f'Ethnicity_{i}', 'value': i} for i in range(4)],
        placeholder="Select Ethnicity",
        value=0
    ),
    dcc.Dropdown(
        id='parental_education',
        options=[{'label': f'Parental Education {i}', 'value': i} for i in range(5)],
        placeholder="Select Parental Education",
        value=0
    ),
    dcc.Dropdown(
        id='parental_support',
        options=[{'label': f'Parental Support {i}', 'value': i} for i in range(5)],
        placeholder="Select Parental Support",
        value=0
    ),
    dcc.Checklist(
        id='sports',
        options=[{'label': 'Sports', 'value': 1}],
        value=[],
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Checklist(
        id='music',
        options=[{'label': 'Music', 'value': 1}],
        value=[],
        labelStyle={'display': 'inline-block'}
    ),
     dcc.Checklist(
        id='tutoring',
        options=[{'label': 'Tutoring', 'value': 1}],
        value=[],
        labelStyle={'display': 'inline-block'}
    ),
     dcc.Checklist(
        id='extracurricular',
        options=[{'label': 'Extracurricular', 'value': 1}],
        value=[],
        labelStyle={'display': 'inline-block'}
    ),
     dcc.Checklist(
        id='volunteering',
        options=[{'label': 'Volunteering', 'value': 1}],
        value=[],
        labelStyle={'display': 'inline-block'}
    ),
    html.Button("Predict", id='predict_button', n_clicks=0),
    html.Div(id='prediction-output')
])

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict_button', 'n_clicks'),
     Input('age', 'value'),
     Input('gender', 'value'),
     Input('study_time', 'value'),
     Input('absences', 'value'),
     Input('tutoring', 'value'),
     Input('extracurricular', 'value'),
     Input('sports', 'value'),
     Input('music', 'value'),
     Input('volunteering', 'value'),
     #Input('gpa', 'value'),
     Input('ethnicity', 'value'),
     Input('parental_education', 'value'),
     Input('parental_support', 'value')]
)
def predict_grade(n_clicks, age, gender,study_time, absences, tutoring,extracurricular, sports, music, volunteering, ethnicity, parental_education, parental_support):
    if n_clicks > 0 and None not in (age, gender, study_time, absences, ethnicity, parental_education, parental_support):
         input_data = {
            'Age': [age],
            'Gender': [gender],
            'Ethnicity': [ethnicity],
            'ParentalEducation': [parental_education],
            'StudyTimeWeekly': [study_time],
            'Absences': [absences],
            'Tutoring': [1 if tutoring and 1 in tutoring else 0],
            'ParentalSupport': [parental_support],
            'Extracurricular': [1 if extracurricular and 1 in extracurricular else 0],
            'Sports': [1 if sports and  1 in sports else 0],
            'Music': [1 if music and 1 in music else 0],
            'Volunteering': [1 if volunteering and 1 in volunteering else 0]
            #'GPA': [gpa],
        }
         
         input_df = pd.DataFrame(input_data)


         input_df = input_df[numerical_cols]
         

         input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
         
    
         #Deep learning
         print("Loading model...")
         DL_model = get_DLmodel()
         print("Model loaded!")

         dl_prediction = DL_model.predict(input_df)
         rf_prediction = randomforest_model.predict(input_df)
         logreg_prediction = regression_model.predict(input_df)
         xgboost_prediction = xgboost_model.predict(input_df)
         print("Prediction done")

         if len(dl_prediction.shape) == 2 and dl_prediction.shape[1] > 1:
             class_prediction = np.argmax(dl_prediction)
             probability = np.max(dl_prediction)
         else:
             class_prediction = int(round(float(dl_prediction[0][0])))
             probability = float(dl_prediction[0][0]) if class_prediction == 1 else 1 - float(dl_prediction[0][0])

         probability_percent = probability * 100
         
        

         return html.Div([
            html.P(f"Deep Learning Prediction: {class_prediction} (Confidence: {probability_percent:.2f}%)"),
            html.P(f"Random Forest Prediction: {rf_prediction[0]}"),
            html.P(f"Logistic Regression Prediction: {logreg_prediction[0]}"),
            html.P(f"XGBoost Prediction: {xgboost_prediction[0]}")
])

    return "Please fill in all fields."
        
if __name__ == "__main__":
    print("Launching Dash app...")
    try:
        app.run(debug=True, host='127.0.0.1', port=8050)
    except Exception as e:
        print("Failed to start server:", e)