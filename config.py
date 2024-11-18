import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_data(df):
    X = df[['marca', 'modelo', 'ano', 'quilometragem', 'cidade', 'estado']]
    y = df['preco']

    X = pd.get_dummies(X, columns=['marca', 'modelo', 'cidade', 'estado'])

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, model_type, params):
    if model_type == 'Decision Tree':
        model = DecisionTreeRegressor(**params)
    elif model_type == 'KNN':
        model = KNeighborsRegressor(**params)
    model.fit(X_train, y_train)
    return model
