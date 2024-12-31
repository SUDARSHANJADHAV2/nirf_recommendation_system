import sys
import os
import pandas as pd
import joblib

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import load_data, preprocess_data, save_processed_data
from feature_engineering import feature_engineering
from model_training import train_model
from recommendation_engine import recommend_improvements

def main():
    try:
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
        joblib.dump(model, "models/random_forest_model.pkl")

        # Step 4: Recommendation Engine
        sample_institution_data = {
            'FSR_value': 70,
            'FQE_PhD_percentage': 60,
            'FQE_experience': 65,
            'LL_library_exp': 55,
            'LL_lab_exp': 60,
            'SEC_sports_area': 50,
            'SEC_sports_budget': 40,
            'PU_publications': 75,
            'CI_citations': 80,
            'IPR_granted': 30,
            'UE_graduation_rate': 85,
            'PE_success_rate': 90,
            'CES_certificates': 45,
            'RD_other_states': 50,
            'RD_foreign_students': 55,
            'WS_women_students': 60,
            'WS_women_faculty': 70,
            'ESDS_percentage': 65,
            'DAP_facilities': 75,
            'PR_peer_ranking': 80,
            'SR_ratio': 85
        }
        sample_institution_df = pd.DataFrame([sample_institution_data])

        # Ensure the data is in the correct format and has valid feature names
        sample_institution_scaled = scaler.transform(sample_institution_df[features])

        # Call recommend_improvements with the correct number of arguments
        predicted_score, recommendations = recommend_improvements(sample_institution_scaled, model, df)

        print(f'Predicted NIRF Score: {predicted_score[0]}')
        print('Recommendations:')
        for rec in recommendations:
            print(f'- {rec}')
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()