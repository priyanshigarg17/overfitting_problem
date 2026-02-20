from sklearn.model_selection import train_test_split
from data_loader import load_data
from preprocessing import scale_features
from feature_selection import select_features
from model import train_model
from evaluate import evaluate

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "high_dim_800_samples_120_features.csv")

print("Looking for file at:", DATA_PATH)

def main():
    # Load data
    X, y = load_data(DATA_PATH)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Feature selection
    X_train_sel, X_test_sel, selector = select_features(
        X_train_scaled, y_train, X_test_scaled
    )

    # Train
    model = train_model(X_train_sel, y_train)

    # Evaluate
    evaluate(model, X_test_sel, y_test)
    
    print("Train Accuracy:", model.score(X_train_sel, y_train))
    print("Test Accuracy:", model.score(X_test_sel, y_test))


if __name__ == "__main__":
    main()