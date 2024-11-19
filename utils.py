import os
from flask import Flask
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
app.secret_key = 'mister-picanha'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    model_path = os.path.join(app.config['MODEL_FOLDER'], 'model.pkl')
    data = joblib.load(model_path)
    return data['model'], data['columns'], data['scaler']

# Função para salvar modelo
def save_model(model, columns, scaler):
    model_path = os.path.join(app.config['MODEL_FOLDER'], 'model.pkl')
    joblib.dump({'model': model, 'columns': columns, 'scaler': scaler}, model_path)
    
# Preparação do dataset
def prepare_dataset(df):
    df = df.dropna()

    df['brand'] = df['name'].str.split().str[0]
    df['age'] = 2024 - df['year']
    df['km_driven'] = np.log1p(df['km_driven'])
    df['selling_price'] = np.log1p(df['selling_price'])

    # Remover a coluna 'name'
    df = df.drop('name', axis=1)

    df = pd.get_dummies(df, columns=['brand', 'fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

    scaler = StandardScaler()
    df[['age', 'km_driven']] = scaler.fit_transform(df[['age', 'km_driven']])
    return df, scaler

