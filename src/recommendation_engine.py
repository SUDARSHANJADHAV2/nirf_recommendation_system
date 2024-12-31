import pandas as pd
import joblib
import numpy as np

def recommend_improvements(sample_institution_scaled, model, df, feature_names, scaler):
    """
    Generate recommendations for improving the NIRF score of an institution.
    
    Parameters:
    sample_institution_scaled (array-like): Scaled feature values of the sample institution.
    model (object): Trained model used for prediction.
    df (DataFrame): Original DataFrame used for training the model.
    feature_names (list): List of feature names used for scaling and training.
    scaler (StandardScaler): Scaler object used to scale the data.
    
    Returns:
    tuple: Predicted score and list of recommendations.
    """
    predicted_score = model.predict(sample_institution_scaled)
    recommendations = []
    
    # Inverse transform the scaled data to get the original values
    sample_institution_unscaled = scaler.inverse_transform(sample_institution_scaled)
    
    for i, feature in enumerate(feature_names):
        avg_value = df[feature].mean()
        if sample_institution_unscaled[0][i] < avg_value:
            recommendations.append(f"Improve {feature} to at least {avg_value:.2f}")

    return predicted_score, recommendations

if __name__ == "__main__":
    # Use an absolute path if necessary
    processed_data_path = "C:/Users/sudar/Desktop/nirf_recommendation_system/data/processed/processed_data.csv"

    df = pd.read_csv(processed_data_path)
    model = joblib.load("C:/Users/sudar/Desktop/nirf_recommendation_system/models/random_forest_model.pkl")
    scaler = joblib.load("C:/Users/sudar/Desktop/nirf_recommendation_system/models/scaler.pkl")

    # Sample institution data
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
        'SR_ratio': 85,
        'PR_peer_ranking': 80
    }
    
    # Ensure the sample data matches the training features
    features = ['FSR_value', 'FQE_PhD_percentage', 'FQE_experience', 
                'LL_library_exp', 'LL_lab_exp', 'SEC_sports_area', 
                'SEC_sports_budget', 'PU_publications', 'CI_citations', 
                'IPR_granted', 'UE_graduation_rate', 'PE_success_rate', 
                'CES_certificates', 'RD_other_states', 'RD_foreign_students', 
                'WS_women_students', 'WS_women_faculty', 'ESDS_percentage', 
                'DAP_facilities', 'SR_ratio', 'PR_peer_ranking']
    
    # Create a DataFrame for the sample institution data with the correct feature names and order
    sample_institution_df = pd.DataFrame([sample_institution_data], columns=features)
    
    # Ensure the order of the columns matches the order used during fitting
    sample_institution_df = sample_institution_df[features]
    
    # Scale the data using the same scaler used during training
    sample_institution_scaled = scaler.transform(sample_institution_df)
    
    # Exclude 'PR_peer_ranking' from the features used for prediction
    features_for_prediction = features[:-1]  # Excluding the last feature 'PR_peer_ranking'
    sample_institution_scaled_for_prediction = sample_institution_scaled[:, :-1]
    
    # Predict the NIRF score and generate recommendations
    predicted_score, recommendations = recommend_improvements(sample_institution_scaled_for_prediction, model, df, features_for_prediction, scaler)

    print(f'Predicted NIRF Score: {predicted_score[0]}')
    print('Recommendations:')
    for rec in recommendations:
        print(f'- {rec}')