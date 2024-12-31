import pandas as pd

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data: handle missing values and clean up."""
    # Fill missing values only for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df

def save_processed_data(df, file_path):
    """Save the processed data to a CSV file."""
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    # Use the absolute path of your dataset
    raw_data_path = "C:/Users/sudar/Desktop/nirf_recommendation_system/data/raw/nirf_2023_indias_national_institutional_ranking.csv"
    processed_data_path = "C:/Users/sudar/Desktop/nirf_recommendation_system/data/processed/processed_data.csv"
    
    df = load_data(raw_data_path)
    df = preprocess_data(df)
    save_processed_data(df, processed_data_path)