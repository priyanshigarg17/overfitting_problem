from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    model = LogisticRegression(penalty='l2')
    model.fit(X_train, y_train)
    return model