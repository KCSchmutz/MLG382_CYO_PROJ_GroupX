#Importing all needed libraries
import pickle
import os
from sklearn.model_selection import train_test_split


# Function to split the dataset dropping column (Objective is to predict profit)
def split_features_target(df, target_column='Profit'):
    X = df.drop(target_column, axis=1)
    Y = df[target_column]
    return X, Y


# Funtion to save feature names (Web App use)
def save_feature_list(features, path):
    with open(path, 'wb') as f:
        pickle.dump(features, f)


# Function to train split data
def create_train_test_split(X, Y, test_size=0.2, random_state=42, save_csv=True):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    #Boolean check to see if train.csv and test.csv already exists
    if save_csv:
        train_path = "../data/train.csv"
        test_path = "../data/test.csv"

        # Only saves if the files don't already exist
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            train_df = X_train.copy()
            train_df["Profit"] = Y_train

            test_df = X_test.copy()
            test_df["Profit"] = Y_test

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

    return X_train, X_test, Y_train, Y_test
