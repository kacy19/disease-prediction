# scripts/preprocessing.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data():
    # Use relative path to dataset
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart_dataset.csv')
    df = pd.read_csv(DATA_PATH)

    cp_columns = [
        'cp_asymptomatic',
        'cp_atypical angina',
        'cp_non-anginal',
        'cp_typical angina'
    ]

    def get_cp_type(row):
        for cp in cp_columns:
            if row[cp] == 1:
                return cp
        return None

    df['cp_type'] = df.apply(get_cp_type, axis=1)
    df.drop(columns=cp_columns, inplace=True)

    for col in ['restecg', 'slope', 'thal']:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    df['exang'] = df['exang'].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})

    le_target = LabelEncoder()
    df['cp_type'] = le_target.fit_transform(df['cp_type'])

    X = df.drop(columns=['cp_type'])
    y = df['cp_type']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42), le_target
