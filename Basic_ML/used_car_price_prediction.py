import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

class ML_trainer:
    
    # load dataset 
    def __init__(self,dataset_path):
        
        self.used_car_df = pd.read_csv(dataset_path,index_col=0)
        
    # function for split_dataset
    def split_data(self,target_col):
        
        self.data_preprocessor()
        self.feature_eng()
        
        self.X = self.used_car_df.drop(target_col,axis=1)
        self.y = self.used_car_df.drop[target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2)
        
        self.data_encoder()
    
    def data_encoder(self):
        # select all catagorical columns
        categorical_columns = self.used_car_df.select_dtypes(include=['object']).columns

        label_encoder = LabelEncoder()

        for column in categorical_columns:
            self.used_car_df[column] = label_encoder.fit_transform(self.used_car_df[column])
            
        self.model()
            
    # function to train model
    def model(self):

        self.linear_model = LinearRegression()
        self.linear_model.fit(self.X_train,self.y_train)
        
        self.evaluation()
        
    # function for model evaluation
    def evaluation(self):
        
        self.y_pred = self.linear_model.predict(self.X_test)
        
        mae = mean_absolute_error(self.y_test,self.y_pred)
        
        print(f"Model score : {self.linear_model.score(self.X_test,self.y_test)}")
        print(f"Model Error : {mae}")    

    # function for data preprocessing
    def data_preprocessor(self):
        
        # remove unnecessary column
        self.used_car_df.drop(["car_name","registration_year","manufacturing_year"],axis=1,inplace=True)
        
        # drop duplicates
        self.used_car_df = used_car_df.drop_duplicates()
        
        self.feature_eng()
        
        #remove outliers
        outlier_cols = ["mileage(kmpl)","engine(cc)","max_power(bhp)","torque(Nm)","price(in lakhs)"]
        self.remove_outliers(outlier_cols)
    
    # function for create new features or to do feature selection    
    def feature_eng(self):
            
        # create some useful feature
        current_year = pd.to_datetime('today').year
        self.used_car_df['manufacturing_year'] = pd.to_numeric(self.used_car_df['manufacturing_year'], errors='coerce').astype('Int64')
        self.used_car_df['Car_Age'] = current_year - self.used_car_df['manufacturing_year']
        self.used_car_df.dropna(inplace=True)
        
    # fucntion for remove outlier    
    def remove_outliers(self,cols):
        
        for col in cols:
            q1 = self.used_car_df[col].quantile(0.25)
            q3 = self.used_car_df[col].quantile(0.75)
            IQR = q3-q1
            
            lower_limit = q1 - IQR*1.5
            upper_limit = q3 + IQR*1.5
            
            self.used_car_df =  self.used_car_df.loc[(self.used_car_df[col]>lower_limit) & (self.used_car_df[col]<upper_limit)]

if __name__ == "__main__":
    
    dataset_path = "Used Car Dataset.csv"
    target_col = "price(in lakhs)"
    trainer = ML_trainer(dataset_path)
    trainer.split_data(target_col)
    