#Importing all required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Function to load data
def load_data(filepath):

    #Loads data into the DataFrame
    df = pd.read_csv(filepath)
    

    #Drop rows and columns that are completely null
    df = df.dropna(how='all')  # Drop rows that are all NaN
    df = df.dropna(axis=1, how='all')  # Drop columns that are all NaN

    #Fill missing numeric values with mean, categorical with mode
    for column in df.columns:

        #Boolean to calculate the sum of all null columns found
        if df[column].isnull().sum() > 0:

            #Boolean to check if column datatype is float64 or int64
            if df[column].dtype in ['float64', 'int64']:

                #Fills columns with mean
                df[column] = df[column].fillna(df[column].mean())
            else:

                #Fills columns with mode
                df[column] = df[column].fillna(df[column].mode()[0])


    # Drops duplicate rows
    df = df.drop_duplicates()


    #Feature Engineering
    #Extract Year, Month, Day, DayofWeek from OrderDate found in df (dataset)
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['Year'] = df['OrderDate'].dt.year
    df['Month'] = df['OrderDate'].dt.month
    df['Day'] = df['OrderDate'].dt.day
    df['DayOfWeek'] = df['OrderDate'].dt.dayofweek

    #Create product cost vs price ratio
    df['CostPriceRatio'] = df['ProductStandardCost'] / df['ProductListPrice']

    #Encode categorical features using one-hot encoding
    df = pd.get_dummies(df, columns=['RegionName', 'CountryName', 'State', 'City', 'CategoryName'], drop_first=True)

    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    df.drop(columns=['OrderDate'], inplace=True)
    
    return df


#Function to remove outliers
def remove_outliers(df, column):

    #Using Interquartile Range (IQR) Approach
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    #Acquiring minimim, maximimum ranges for each column
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    #Returns the DataFrame to keep only the rows values in the IQR range
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


#Function that loops over and over to remove all outliers (Not all outliers are removed after one iteration)
def iterative_outlier_removal(df, numerical_columns):

    #While loop that continues looping to remove all outliers
    while True:

        #Variable prev_shape to holp the dataframe's shape
        prev_shape = df.shape

        #For loop to iterate over each numerical column
        for col in numerical_columns:

            #This runs the remove outliers function above
            df = remove_outliers(df, col)

        #Boolean check to see if the new shape is the same as the previous shape
        if df.shape == prev_shape:

            #If it is the loop breaks signaling that there are no more outliers
            break

    return df


#Function to scale features using StandardScaler
def scale_features(df, numerical_columns):

    #Initialization of StandardScaler
    scaler = StandardScaler()

    #Transforming the numerical columns in the dataframe with scaling
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    #Scaler is also returned in case new data needs to be fitted
    return df, scaler
