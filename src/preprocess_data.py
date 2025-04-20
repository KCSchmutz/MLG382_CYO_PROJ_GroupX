#Importing all required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re

#Function to load and preprocess data
def load_data(filepath):
    #Load CSV
    df = pd.read_csv(filepath)

    #Clean string columns to remove extra spaces, tabs, line breaks
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x.strip()))

    #Drop rows and columns that are completely null
    df = df.dropna(how='all')  # Drop all-NaN rows
    df = df.dropna(axis=1, how='all')  # Drop all-NaN columns

    #Fill missing values
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if df[column].dtype in ['float64', 'int64']:
                df[column] = df[column].fillna(df[column].mean())
            else:
                df[column] = df[column].fillna(df[column].mode()[0])

    #Drop duplicates
    df = df.drop_duplicates()

    # Feature Engineering: Extract date parts
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['Year'] = df['OrderDate'].dt.year
    df['Month'] = df['OrderDate'].dt.month
    df['Day'] = df['OrderDate'].dt.day
    df['DayOfWeek'] = df['OrderDate'].dt.dayofweek

    #Derived metric: cost vs price
    df['CostPriceRatio'] = df['ProductStandardCost'] / df['ProductListPrice']

    #One-hot encode key categorical features
    df = pd.get_dummies(df, columns=['RegionName', 'CountryName', 'State', 'City', 'CategoryName'], drop_first=True)

    #Label encode remaining object columns (if any)
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    #Drop original date column
    df.drop(columns=['OrderDate'], inplace=True)

    return df

#Function to remove outliers using IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

#Function to iteratively remove outliers from multiple columns
def iterative_outlier_removal(df, numerical_columns):
    while True:
        prev_shape = df.shape
        for col in numerical_columns:
            df = remove_outliers(df, col)
        if df.shape == prev_shape:
            break
    return df

#Standard scaling for numerical columns
def scale_features(df, numerical_columns):
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df, scaler
