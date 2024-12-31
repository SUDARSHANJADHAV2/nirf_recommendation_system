import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def train_model(X, y):
    """Train a RandomForest model on the dataset."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    return model

if __name__ == "__main__":
    processed_data_path = "C:/Users/sudar/Desktop/nirf_recommendation_system/data/processed/processed_data.csv"
    model_path = "C:/Users/sudar/Desktop/nirf_recommendation_system/models/trained_model.pkl"
    
    df = load_data(processed_data_path)
    features = ['FSR_value', 'FQE_PhD_percentage', 'FQE_experience', 
                'LL_library_exp', 'LL_lab_exp', 'SEC_sports_area', 
                'SEC_sports_budget', 'PU_publications', 'CI_citations', 
                'IPR_granted', 'UE_graduation_rate', 'PE_success_rate', 
                'CES_certificates', 'RD_other_states', 'RD_foreign_students', 
                'WS_women_students', 'WS_women_faculty', 'ESDS_percentage', 
                'DAP_facilities', 'PR_peer_ranking', 'SR_ratio']
    X = df[features]
    y = df['PR_peer_ranking']  # Assuming 'PR_peer_ranking' is the target variable
    model = train_model(X, y)
    
    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Trained model saved to {model_path}")