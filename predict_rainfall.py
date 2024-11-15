import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load and preprocess data
def load_data():
    data = pd.read_csv('data/rainfall_data.csv')
    data.dropna(inplace=True)  # Handle missing data
    return data

# Preprocess features and labels
def preprocess_data(data):
    # Convert months to numerical features if necessary
    features = data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']]
    labels = data['ANNUAL']
    return features, labels

# Train model
def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model evaluation
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
    print(f"R^2 Score: {r2_score(y_test, y_pred)}")

    joblib.dump(model, 'rainfall_model.pkl')
    print("Model saved as rainfall_model.pkl")

if __name__ == "__main__":
    data = load_data()
    features, labels = preprocess_data(data)
    train_model(features, labels)
