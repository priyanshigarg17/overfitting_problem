from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

def select_features(X_train, y_train, X_test):
    model = LogisticRegression(penalty='l2', solver='liblinear')
    selector = SelectFromModel(model)
    
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    return X_train_selected, X_test_selected, selector