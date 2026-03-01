import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_risk_engine():
    print("Loading 1 year of historical grid data (525,600 rows)...")
    try:
        df = pd.read_csv('historical_grid_data.csv')
    except FileNotFoundError:
        print("ERROR: historical_grid_data.csv not found! Ensure it's in the same folder.")
        return

    # 1. Feature Engineering: What factors cause transmission losses?
    features = ['Load_Amps', 'Ambient_Temp', 'Line_Length_km']
    target = 'Loss_Percentage'

    X = df[features]
    y = df[target]

    # 2. Split data into 80% Training and 20% Testing (Standard ML Practice)
    print("Splitting data into 80% Training and 20% Testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train the AI Model
    print("Training the AI (Random Forest Regressor)...")
    print("Using all CPU cores to crunch 500k+ rows. This may take 30-60 seconds...")
    
    # n_jobs=-1 tells the computer to use all processors for maximum speed
    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 4. Evaluate AI Performance
    print("\nEvaluating AI Performance on unseen test data...")
    predictions = model.predict(X_test)
    accuracy = r2_score(y_test, predictions)
    
    print("\n========================================")
    print("       AI MODEL PERFORMANCE STATUS      ")
    print("========================================")
    print(f"Model Accuracy (R-squared) : {accuracy * 100:.2f}%")
    
    if accuracy > 0.85:
        print("Status: EXCELLENT. The AI successfully learned the grid physics.")
    else:
        print("Status: OK, but might need tuning.")

    # 5. The "Pitch Winning" Metric: Feature Importances
    importances = model.feature_importances_
    print("\n[FOR THE PITCH DECK] Feature Importances:")
    print("Tell the judges exactly what causes the most losses:")
    for feature, imp in zip(features, importances):
        print(f"- {feature}: {imp * 100:.1f}%")

    # 6. Save the model for the Streamlit Dashboard
    model_filename = 'risk_model.pkl'
    joblib.dump(model, model_filename)
    print(f"\nSuccess! AI Model compiled and saved as '{model_filename}'")
    print("Hand this .pkl file to Person 3 for the UI!")

if __name__ == "__main__":
    train_risk_engine()