import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# function to train model parameter : dataset,target column
def model(df,target_col):
        
    # select all catagorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    label_encoder = LabelEncoder()

    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    
    # divide dataset into X and y    
    X = df.drop(target_col,axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    linear_model = LinearRegression()
    linear_model.fit(X_train,y_train)
    
    # make prediction
    y_pred = linear_model.predict(X_test)
    
    mae = mean_absolute_error(y_test,y_pred)
    
    # return model score and error
    return [linear_model.score(X_test,y_test),mae]

def data_preprocessor(used_car_df):
    
    # remove unnecessary column
    used_car_df.drop(["car_name","registration_year","manufacturing_year"],axis=1,inplace=True)
    # drop duplicates
    used_car_df = used_car_df.drop_duplicates()
    
    #remove outliers
    used_car_df = remove_outliers("mileage(kmpl)")
    used_car_df = remove_outliers("engine(cc)")
    used_car_df = remove_outliers("max_power(bhp)")
    used_car_df = remove_outliers("torque(Nm)")
    used_car_df = remove_outliers("price(in lakhs)")
    
    # create some useful feature
    current_year = pd.to_datetime('today').year
    used_car_df['manufacturing_year'] = pd.to_numeric(used_car_df['manufacturing_year'], errors='coerce').astype('Int64')
    used_car_df['Car_Age'] = current_year - used_car_df['manufacturing_year']
    used_car_df.dropna(inplace=True)
    
# fucntion for remove outlier    
def remove_outliers(col):
    q1 = used_car_df[col].quantile(0.25)
    q3 = used_car_df[col].quantile(0.75)
    IQR = q3-q1
    
    lower_limit = q1 - IQR*1.5
    upper_limit = q3 + IQR*1.5
    
    return used_car_df.loc[(used_car_df[col]>lower_limit) & (used_car_df[col]<upper_limit)]

used_car_df = pd.read_csv("Used Car Dataset.csv",index_col=0)
target_col = "price(in lakhs)"

data_preprocessor(used_car_df)
score,mae = model(used_car_df,target_col)
print(score,mae)
