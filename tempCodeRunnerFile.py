from src.data_processing import load_data, preprocess_data, save_processed_data
from src.feature_engineering import feature_engineering
from src.model_training import train_model
from src.recommendation_engine import recommend_improvements
import pandas as pd
import joblib

def main():
    # Step 1: Data Preprocessing
    raw_data_path = "data/raw/nirf_2023_indias_national_institutional_ranking.csv"
    processed_data_path = "data/processed/processed_data.csv"
    df = load_data(raw_data_path)
    df = preprocess_data(df)
    save_processed_data(df, processed_data_path)
    
    # Step 2: Feature Engineering
    df, scaler = feature_engineering(df)
    save_processed_data(df, processed_data_path)
    joblib.dump(scaler, "models/scaler.pkl")
    
    # Step 3: Model Training
    features = ['Teaching_Learning_Resources', 'Research_Professional_Practice', 
                'Graduation_Outcomes', 'Outreach_Inclusivity', 'Perception']
    X = df[features]
    y = df['Score']
    model = train_model(X, y)
    joblib.dump(model, "models/random_forest_model.pkl")
    
    # Step 4: Recommendation Engine
    sample_institution_data = {
        'Teaching_Learning_Resources': 70,
        'Research_Professional_Practice': 60,
        'Graduation_Outcomes': 65,
        'Outreach_Inclusivity': 55,
        'Perception': 60
    }
    predicted_score, recommendations = recommend_improvements(sample_institution_data, model, scaler, df)
    print(f'Predicted NIRF Score: {predicted_score[0]}')
    print('Recommendations:')
    for rec in recommendations:
        print(f'- {rec}')

if __name__ == "__main__":
    main()