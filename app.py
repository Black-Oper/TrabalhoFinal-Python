from flask import request, render_template, redirect, url_for, flash
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import app, allowed_file, load_model, save_model, prepare_dataset, generate_interactive_plots
from Forms import UploadForm, ModelForm, PredictionForm

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)


# Página de upload
@app.route('/', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        
        file = form.file.data
        
        if file and allowed_file(file.filename):
            filename = 'car_data.csv'
            
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            flash('Upload realizado com sucesso!')
            return redirect(url_for('analysis'))
        
        else:
            flash('Arquivo inválido. Apenas arquivos CSV são permitidos.')

    return render_template('upload.html', form=form)

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'car_data.csv')

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        flash('Nenhum arquivo carregado. Faça upload dos dados.')
        return redirect(url_for('upload'))

    form = ModelForm()
    metrics = None
    model_trained = False

    brand_chart, year_chart = generate_interactive_plots(df)

    df, scaler = prepare_dataset(df)
    X = df.drop('selling_price', axis=1)
    y = df['selling_price']

    if form.validate_on_submit():
        model_type = form.model_type.data
        param = form.param.data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == 'DecisionTree':
            model = DecisionTreeRegressor(max_depth=param, random_state=42)
        elif model_type == 'RandomForest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=param, random_state=42)
        elif model_type == 'GradientBoosting':
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=param, random_state=42)
        else:
            flash('Tipo de modelo inválido.')
            return redirect(request.url)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        metrics = {
            'R2': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'CV R2 Mean': cv_scores.mean()
        }

        save_model(model, X.columns.tolist(), scaler)
        model_trained = True

    return render_template('analysis.html', form=form, metrics=metrics, model_trained=model_trained,
                           brand_chart=brand_chart, year_chart=year_chart)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = PredictionForm()

    try:
        model, model_columns, scaler = load_model()
    except FileNotFoundError:
        flash('Modelo não encontrado. Por favor, treine o modelo primeiro.')
        return redirect(url_for('analysis'))

    brand_columns = [col for col in model_columns if col.startswith('brand_')]
    brands = [col.replace('brand_', '') for col in brand_columns]
    form.brand.choices = [(brand, brand) for brand in brands]

    prediction = None

    if form.validate_on_submit():
        # Construir input data
        input_dict = {}
        for col in model_columns:
            input_dict[col] = 0

        # Definir características categóricas
        categorical_mappings = {
            'fuel_' + form.fuel.data: 1,
            'transmission_' + form.transmission.data: 1,
            'seller_type_' + form.seller_type.data: 1,
            'owner_' + form.owner.data: 1,
            'brand_' + form.brand.data: 1
        }

        for key, value in categorical_mappings.items():
            if key in input_dict:
                input_dict[key] = value

        # Definir características numéricas
        age = form.age.data
        km_driven = form.km_driven.data

        numerical_features = pd.DataFrame({'age': [age], 'km_driven': [np.log1p(km_driven)]})
        numerical_features_scaled = scaler.transform(numerical_features)

        input_dict['age'] = numerical_features_scaled[0][0]
        input_dict['km_driven'] = numerical_features_scaled[0][1]

        input_df = pd.DataFrame([input_dict], columns=model_columns)

        log_prediction = model.predict(input_df)
        prediction = np.expm1(log_prediction[0])

    return render_template('predict.html', form=form, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
