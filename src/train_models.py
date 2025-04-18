#Importing all required libraries
import importlib
import pickle 
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping



#Function to reload the module and train the model
def reload_and_train(X_train, X_test, Y_train, Y_test, model_type="random_forest"):

    #Reload the module to reflect changes (Don't have to restart Kernel everytime if this is included)
    import train_models
    importlib.reload(train_models)
    

    #Train the model based on the model_type specified
    if model_type == "random_forest":

        #Calls the function to train a RandomForestRegressor model
        train_models.train_random_forest(X_train, X_test, Y_train, Y_test)

    elif model_type == "gradient_boosting":

        #Calls the function to train a RandomForestRegressor model
        train_models.train_gradient_boosting(X_train, X_test, Y_train, Y_test)

    elif model_type == "adaboost":

        #Calls the function to train a RandomForestRegressor model
        train_models.train_adaboost(X_train, X_test, Y_train, Y_test)

    elif model_type == "xgboost":

        #Calls the function to train a RandomForestRegressor model
        train_models.train_xgboost(X_train, X_test, Y_train, Y_test)

    elif model_type == 'deep_learning':

        #Adds training and test sets together for deep learning training
        X = pd.concat([X_train, X_test], axis=0)
        Y = pd.concat([Y_train, Y_test], axis=0)

        #Calls the function to train a deep learning model
        train_models.train_deep_learning_model(X,Y)


#Function to create a preprocessor for categorical features 
def get_preprocessor(X):

    #Identify categorical columns (columns with datatype of 'object' and 'category')
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    #Creates a preprocessing pipeline for categorical columns with OneHotEncoder (Changes categorical data into numerical data)
    preprocessor = ColumnTransformer(
        transformers=[ 
            ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        
        #Leave numerical columns unchanged in the output
        remainder='passthrough'  
    )
    return preprocessor


#Function that trains, evaluates, saves and visualizes models
def train_model(model, model_name, X_train, X_test, Y_train, Y_test, output_dir="../artifacts"):

    #Creating preprocessor for categorical features
    preprocessor = get_preprocessor(X_train)

    #Adding preprocessing model into one pipeline
    pipeline = make_pipeline(preprocessor, model)

    #Trains entire pipeline
    pipeline.fit(X_train, Y_train)

    #Predicts test set results
    Y_pred = pipeline.predict(X_test)

    #Prints evaluatoin metrics for each model
    print(f"Model: {model_name}")

    #Used to measure how well predictions match actual values
    print(f"R2 Score: {r2_score(Y_test, Y_pred):.4f}") 

    #Prints average magnitude of prediction error
    print(f"Mean Absolute Error: {mean_absolute_error(Y_test, Y_pred):.2f}")

    #Prints large errors
    print(f"Mean Squared Error: {mean_squared_error(Y_test, Y_pred):.2f}")

    #Prints spacing between metrics and graphs
    print("-" * 50)

    #Save a model to a specific pkl file
    filepath = os.path.join(output_dir, f"{model_name}_model.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved model to: {filepath}")

    #Save predictions
    results_df = pd.DataFrame({
        "Model": model_name,
        "Actual": Y_test.values,
        "Predicted": Y_pred
    })

    predictions_path = "../artifacts/predictions.csv"
    if os.path.exists(predictions_path):
        results_df.to_csv(predictions_path, mode='a', index=False, header=False)
    else:
        results_df.to_csv(predictions_path, index=False)

    #Scatter plot (Shows Actual Values against Predicted Values)
    plt.figure(figsize=(6, 4))
    plt.scatter(Y_test, Y_pred, alpha=0.7)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.title(f"{model_name} - Predicted vs Actual")
    plt.tight_layout()
    plt.show()

    #Line plot (Shows Actual Values against Predicted Values)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(Y_test)), Y_test, label="Actual", marker='o')
    plt.plot(range(len(Y_pred)), Y_pred, label="Predicted", marker='x')
    plt.title(f"{model_name} - Predicted vs Actual")
    plt.xlabel("Count")
    plt.ylabel("Profit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#Function that trains RandomForestRegressor model
def train_random_forest(X_train, X_test, Y_train, Y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    train_model(model, "RandomForestRegressor", X_train, X_test, Y_train, Y_test)
    

#Function that trains GradientBoostingRegressor model
def train_gradient_boosting(X_train, X_test, Y_train, Y_test):
    model = GradientBoostingRegressor(random_state=42)
    train_model(model, "GradientBoostingRegressor", X_train, X_test, Y_train, Y_test)

#Function that trains AdaBoostRegressor model
def train_adaboost(X_train, X_test, Y_train, Y_test):
    model = AdaBoostRegressor(random_state=42)
    train_model(model, "AdaBoostRegressor", X_train, X_test, Y_train, Y_test)

#Function that trains XGBRegressor model
def train_xgboost(X_train, X_test, Y_train, Y_test):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    train_model(model, "XGBRegressor", X_train, X_test, Y_train, Y_test)

#Function that trains Deep learning model
def train_deep_learning_model(X_train, X_test, Y_train, Y_test, model_name="DeepLearningRegressor", output_dir="../artifacts"):
    
    # Build a simple feedforward neural network (Works with layers input, hidden and output)
    model = Sequential()

    #First hidden layer
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    
    #Reduces overfitting the data
    model.add(Dropout(0.3))
    
    #Second Hidden Layer
    model.add(Dense(64, activation='relu'))
    
    #Reduces overfitting data again
    model.add(Dropout(0.2))

    #Output layer
    model.add(Dense(1))

    #Mean Squared Error loss and Adam Optimizer for better convergence
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    #Early stopping to avoid overfitting (if no improvement is found)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    #Trains the model and stores training history for plotting graphs
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    #Gets predictions and flattens them 
    Y_pred = model.predict(X_test).flatten()

    #Evaluates performance metrics
    print(f"Model: {model_name}")
    print(f"R2 Score: {r2_score(Y_test, Y_pred):.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error(Y_test, Y_pred):.2f}")
    print(f"Mean Squared Error: {mean_squared_error(Y_test, Y_pred):.2f}")
    print("-" * 50)

    #Saves the model
    model_path = os.path.join(output_dir, f"{model_name}.h5")
    save_model(model, model_path)
    print(f"Saved model to: {model_path}")

    #Scatter plot: (plots actual values vs predicted values)
    plt.figure(figsize=(6, 4))
    plt.scatter(Y_test, Y_pred, alpha=0.7)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.title(f"{model_name} - Predicted vs Actual")
    plt.tight_layout()
    plt.show()


    #Line Graph: (plots actual values vs predicted values)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(Y_test)), Y_test, label='Actual', marker='o')
    plt.plot(range(len(Y_pred)), Y_pred, label='Predicted', marker='x')
    plt.title(f'{model_name} - Predicted vs Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('Profit')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Loss Graph: (plots training loss vs validation loss to check overfitting or underfitting)
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{model_name} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model

