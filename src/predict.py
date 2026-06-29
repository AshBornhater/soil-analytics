import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

def load_data(base_dir):
    train_path = os.path.join(base_dir, "data", "raw", "train.csv")
    test_path = os.path.join(base_dir, "data", "raw", "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Data hilang di data/raw/.")
        
    return pd.read_csv(train_path), pd.read_csv(test_path)

def preprocess_and_engineer_features(df_train, df_test):
    # Split fitur & target
    X_train = df_train.drop(columns=['sample_id', 'property_organic_content']).copy()
    y_train = df_train['property_organic_content'].copy()
    X_test = df_test.drop(columns=['sample_id']).copy()
    
    # 1. Fillna spektral pakai 0.0
    kolom_pc = [col for col in X_train.columns if '_PC_' in col]
    X_train[kolom_pc] = X_train[kolom_pc].fillna(0.0)
    X_test[kolom_pc] = X_test[kolom_pc].fillna(0.0)
    
    # 2. Fillna numerik sisa pakai median train
    kolom_num = X_train.select_dtypes(include=['int64', 'float64']).columns
    kolom_num_sisa = [col for col in kolom_num if '_PC_' not in col]
    
    median_values = X_train[kolom_num_sisa].median()
    X_train[kolom_num_sisa] = X_train[kolom_num_sisa].fillna(median_values)
    X_test[kolom_num_sisa] = X_test[kolom_num_sisa].fillna(median_values)
    
    # 3. Fitur rasio tekstur tanah
    X_train['ratio_fine_coarse'] = X_train['property_particle_fine'] / (X_train['property_particle_coarse'] + 1)
    X_test['ratio_fine_coarse'] = X_test['property_particle_fine'] / (X_test['property_particle_coarse'] + 1)
    
    # 4. One-Hot Encoding
    kolom_kat = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    encoded_train = encoder.fit_transform(X_train[kolom_kat])
    encoded_test = encoder.transform(X_test[kolom_kat])
    encoded_cols = encoder.get_feature_names_out(kolom_kat)
    
    df_enc_train = pd.DataFrame(encoded_train, columns=encoded_cols, index=X_train.index)
    df_enc_test = pd.DataFrame(encoded_test, columns=encoded_cols, index=X_test.index)
    
    # Gabung fitur & drop kolom teks asli
    X_train_final = pd.concat([X_train.drop(columns=kolom_kat), df_enc_train], axis=1)
    X_test_final = pd.concat([X_test.drop(columns=kolom_kat), df_enc_test], axis=1)
    
    return X_train_final, y_train, X_test_final

def train_and_predict(X_train, y_train, X_test, df_test):
    # Fit model baseline terbaik
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Format df submisi
    submission = pd.DataFrame({
        'sample_id': df_test['sample_id'],
        'property_organic_content': predictions
    })
    return submission

if __name__ == "__main__":
    # Setup root dir proyek
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    
    try:
        # Run pipeline
        df_train, df_test = load_data(BASE_PROJECT_DIR)
        X_train, y_train, X_test = preprocess_and_engineer_features(df_train, df_test)
        submission_df = train_and_predict(X_train, y_train, X_test, df_test)
        
        # Export CSV
        output_dir = os.path.join(BASE_PROJECT_DIR, "data", "processed")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "submission_final2.csv")
        
        submission_df.to_csv(output_path, index=False)
        print(f"Sukses! Output di: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")