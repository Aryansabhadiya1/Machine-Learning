from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_breast_cancer  # load dataset

# we build class to train our model
class ML_trainer:
    
    # # load dataset 
    # def __init__(self,dataset):
        
    #     self.df = pd.read_csv(dataset)
    
    # function for split_dataset
    def split_data(self): # split_data(self,target_col)
        
        # self.X = df.drop(target_col,axis=1)
        # self.y = df.drop[target_col]
        self.X,self.y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=0.2)
        
        self.model()
    
    # function for train model
    def model(self):

        lr_model = LogisticRegression()

        # build dictionary of parameter to do hypertunnig
        param_grid = {"max_iter":[1000,1500,1800]}

        # find best model
        grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='accuracy')

        # train model
        grid_search.fit(self.X_train,self.y_train)
        
        self.evaluation(grid_search)    
    
    # function for evaluate model
    def evaluation(self,model):
        
        #predict value
        self.y_pred = model.predict(self.X_test)

        #make report of perfomance
        cr = classification_report(self.y_test,self.y_pred)
        
        print(cr)
    
    
if __name__ == "__main__":
    
    # Here we are create object of class for train our model
    # Also we can give dataset path as argument for load dataset
    trainer = ML_trainer() # ML_trainer(dataset_path)
    
    # In this method we will split training & testing data and after that all process is automatic
    # We can also give our target_column here as argument to split but here i use inbuild sklearn data so I can't implement it.
    trainer.split_data() # split_data("IsCancer")