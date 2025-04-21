import os
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pickle
from tensorflow.keras.models import load_model

# Load model artifacts
model_paths = {
    "AdaBoost": "../artifacts/AdaBoostRegressor_model.pkl",
    "GradientBoosting": "../artifacts/GradientBoostingRegressor_model.pkl",
    "RandomForest": "../artifacts/RandomForestRegressor_model.pkl",
    "XGBoost": "../artifacts/XGBRegressor_model.pkl",
}

models = {}
for name, path in model_paths.items():
    with open(path, 'rb') as f:
        models[name] = pickle.load(f)

# Load deep learning model without compiling (for inference only)
DL_model = load_model("../artifacts/DeepLearningRegressor.h5", compile=False)

# Load reference data to infer input features
df_ref = pd.read_csv("../artifacts/processed_data.csv")
if 'TotalItemQuantity' in df_ref.columns:
    df_ref.drop(columns=['TotalItemQuantity'], inplace=True)

all_features = df_ref.columns.tolist()
numerical_features = [col for col in df_ref.columns if df_ref[col].nunique() > 10 and not any(prefix in col for prefix in ['RegionName_', 'CountryName_', 'State_', 'City_', 'CategoryName_'])]
categorical_features = [col for col in df_ref.columns if col not in numerical_features]

# Get default values from the first row of the dataset
default_values = df_ref.iloc[0].to_dict()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H2("Total Item Quantity Prediction Tool", className="text-center text-primary my-4"),

    dbc.Row([
        dbc.Col([
            dbc.Label(field),
            dbc.Input(id=field, type='number', value=default_values.get(field, 0))
        ], width=4) for field in numerical_features
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Select Model"),
            dcc.Dropdown(
                id='model_selector',
                options=[
                    {'label': model_name, 'value': model_name}
                    for model_name in ["Deep Learning", "RandomForest", "AdaBoost", "GradientBoosting", "XGBoost"]
                ],
                value="Random Forest"
            )
        ], width=6),
    ], className="mb-4"),

    dbc.Button("Predict Total Item Quantity", id='predict_button', color="primary", className="mb-3"),
    html.Div(id='prediction-output')
])

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict_button', 'n_clicks'),
    State('model_selector', 'value'),
    [State(field, 'value') for field in numerical_features]
)
def predict_totalitemquantity(n_clicks, selected_model, *inputs):
    if n_clicks:
        input_data = dict(zip(numerical_features, inputs))
        df = pd.DataFrame([input_data])

        # Add zero columns for all other features
        for col in all_features:
            if col not in df.columns:
                df[col] = 0

        # Reorder to match model input
        df = df[all_features]

        # Predict using the selected model
        if selected_model == "Deep Learning":
            pred = DL_model.predict(df)[0][0]
        else:
            pred = models[selected_model].predict(df)[0]

        return html.Div([
            dbc.Card([
                dbc.CardHeader("Prediction Result"),
                dbc.CardBody([
                    html.H5(f"{selected_model} Prediction: ${pred:.2f}")
                ])
            ], color="light"),
            html.Div([
                html.Hr(),
                dcc.Graph(
                    figure={
                        'data': [
                            {'x': list(range(1, 2)), 'y': [pred], 'type': 'line', 'name': selected_model}
                        ],
                        'layout': {
                            'title': f'{selected_model} Prediction Visualization',
                            'xaxis': {'title': 'Prediction Index'},
                            'yaxis': {'title': 'Predicted Total Item Quantity'}
                        }
                    }
                )
            ])
        ])

    return "Please fill in all fields."

if __name__ == '__main__':
    app.run(debug=True)
