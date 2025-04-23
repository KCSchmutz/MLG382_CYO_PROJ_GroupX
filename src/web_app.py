import os
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pickle
from tensorflow.keras.models import load_model

# Load model artifacts
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root of repo

model_paths = {
    "AdaBoost": os.path.join(base_dir, "artifacts", "AdaBoostRegressor_model.pkl"),
    "GradientBoosting": os.path.join(base_dir, "artifacts", "GradientBoostingRegressor_model.pkl"),
    "RandomForest": os.path.join(base_dir, "artifacts", "RandomForestRegressor_model.pkl"),
    "XGBoost": os.path.join(base_dir, "artifacts", "XGBRegressor_model.pkl"),
}


models = {}
for name, path in model_paths.items():
    with open(path, 'rb') as f:
        models[name] = pickle.load(f)

DL_model = load_model(os.path.join(base_dir, "artifacts", "DeepLearningRegressor.h5"), compile=False)
df_ref = pd.read_csv(os.path.join(base_dir, "artifacts", "processed_data.csv"))

if 'TotalItemQuantity' in df_ref.columns:
    df_ref.drop(columns=['TotalItemQuantity'], inplace=True)

all_features = df_ref.columns.tolist()
numerical_features = [col for col in df_ref.columns if df_ref[col].nunique() > 10 and not any(prefix in col for prefix in ['RegionName_', 'CountryName_', 'State_', 'City_', 'CategoryName_'])]
categorical_features = [col for col in df_ref.columns if col not in numerical_features]

# Get default values from the first row of the dataset
default_values = df_ref.iloc[0].to_dict()

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

COLORS = {
    "primary": "#1a1b41",      # Navy blue
    "background": "#fdfffc",   # Off-white
    "accent": "#c1c9d9",       # Optional: subtle highlight
    "text": "#1a1b41",         # Match primary for dark text
    "card_bg": "#ffffff"       # Clean white cards
}


app.layout = dbc.Container([
    dbc.Card([
        dbc.CardBody([

            html.H2("Total Item Quantity Prediction Tool", className="text-center", style={
                "color": COLORS["primary"], "marginTop": "30px", "marginBottom": "30px"
            }),

            dbc.Row([
                dbc.Col([

                    # Product Details
                    dbc.Card([
                        dbc.CardHeader("Product Details", style={
                            "backgroundColor": COLORS["primary"], "color": COLORS["background"]
                        }),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label(field, style={"color": COLORS["text"]}),
                                    dbc.Input(id=field, type='number', value=default_values.get(field, 0), className="mb-3")
                                ]) for field in numerical_features if 'Product' in field or 'Category' in field
                            ])
                        ])
                    ], className="mb-4", style={"backgroundColor": COLORS["card_bg"]}),

                    # Order Details
                    dbc.Card([
                        dbc.CardHeader("Order Details", style={
                            "backgroundColor": COLORS["primary"], "color": COLORS["background"]
                        }),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label(field, style={"color": COLORS["text"]}),
                                    dbc.Input(id=field, type='number', value=default_values.get(field, 0), className="mb-3")
                                ]) for field in numerical_features if 'Order' in field or 'Month' in field or 'Day' in field or field not in ['Product', 'Category']
                            ])
                        ])
                    ], className="mb-4", style={"backgroundColor": COLORS["card_bg"]}),

                    # Model Selection
                    dbc.Card([
                        dbc.CardHeader("Model Selection", style={
                            "backgroundColor": COLORS["primary"], "color": COLORS["background"]
                        }),
                        dbc.CardBody([
                            dbc.Label("Select Model", style={"color": COLORS["text"]}),
                            dcc.Dropdown(
                                id='model_selector',
                                options=[{'label': name, 'value': name} for name in ["Deep Learning", "RandomForest", "AdaBoost", "GradientBoosting", "XGBoost"]],
                                value="RandomForest",
                                className="mb-3"
                            ),
                            dbc.Button("Predict Total Item Quantity", id='predict_button', color="dark", className="w-100")
                        ])
                    ], className="mb-4", style={"backgroundColor": COLORS["card_bg"]}),

                    html.Div(id='prediction-output')

                ], width=8, className="mx-auto")
            ])

        ])
    ], style={
        "padding": "30px",
        "marginTop": "40px",
        "boxShadow": "0 4px 15px rgba(0, 0, 0, 0.1)",
        "borderRadius": "15px",
        "backgroundColor": "#ffffff"
    })
], fluid=True, style={
    "backgroundColor": COLORS["background"],
    "paddingBottom": "50px",
    "paddingTop": "30px"
})



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
                    html.H5(f"{selected_model} Prediction: {pred:.2f} Total Item Quantity")
                ])
            ], color="light"),
            html.Div([
                html.Hr()
            ])
        ])

    return "Please fill in all fields."

if __name__ == '__main__':
    app.run(debug=True)
