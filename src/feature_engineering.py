from sklearn.preprocessing import StandardScaler
from data_processing import load_data, preprocess_data

def feature_engineering(df):
    """Perform feature engineering: scaling numerical features."""
    features = [
        'FSR_value', 'FQE_PhD_percentage', 'FQE_experience', 'LL_library_exp', 
        'LL_lab_exp', 'SEC_sports_area', 'SEC_sports_budget', 'PU_publications', 
        'CI_citations', 'IPR_granted', 'UE_graduation_rate', 'PE_success_rate', 
        'CES_certificates', 'RD_other_states', 'RD_foreign_students', 'WS_women_students', 
        'WS_women_faculty', 'ESDS_percentage', 'DAP_facilities', 'PR_peer_ranking', 'SR_ratio'
    ]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler

if __name__ == "__main__":
    processed_data_path = "C:/Users/sudar/Desktop/nirf_recommendation_system/data/processed/processed_data.csv"
    df = load_data(processed_data_path)
    df = preprocess_data(df)
    df, scaler = feature_engineering(df)
    df.to_csv(processed_data_path, index=False)