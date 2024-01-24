# breast_cancer.py

--> In this project I use breast_cancer dataset of sklearn to build a model that predicts whether a person has breast cancer or not.
I use LogisticRegression for modeling and do hyperparameter tunning using GridsearchCV.
I build a class for all the steps to perform for the training model.
I add methods in class like split_data, model, evaluation, etc.
In this format, you have to only make the object of the class and call split_data method and done all things like pipeline.
Model accuracy is around 0.96.

# used_car_price_pridiction.py

--> In this project I use used car data to predict the price of it. In that, I made functions for remove_outlier, data preprocessing, and model building and added comments when necessary.
Ans I use the linearRegression model to train despite it underfitting the data but the main intention of preparing this project is to build an understanding of coding structure or  project structure
I use a project structure the same as breast_cancer.py but in this, I use an external file to make it fully automatic.
In this you have to make an object of the class and give the dataset path as an argument in class to load data and call split_data method which first does data preprocessing and feature engineering and after that do modeling.
