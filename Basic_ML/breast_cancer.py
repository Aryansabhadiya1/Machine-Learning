from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_breast_cancer  # load dataset


def model(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    lr_model = LogisticRegression()

    # build dictionary of parameter to do hypertunnig
    param_grid = {"max_iter":[1000,1500,1800]}

    # find best model
    grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='accuracy')

    # train model
    grid_search.fit(X_train,y_train)

    #predict value
    y_pred = grid_search.predict(X_test)

    #make report of perfomance
    cr = classification_report(y_test,y_pred)
    
    return cr
    
# load data in x and y
X,y = load_breast_cancer(return_X_y=True)

cr = model(X,y)

print(cr)