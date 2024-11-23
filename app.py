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

# Configurações da aplicação Flask
app.config['UPLOAD_FOLDER'] = 'uploads'  # Diretório para armazenar uploads de arquivos
app.config['MODEL_FOLDER'] = 'models'  # Diretório para armazenar modelos treinados
app.config['ALLOWED_EXTENSIONS'] = {'csv'}  # Tipos de arquivos permitidos

# Garante que os diretórios necessários existam
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Rota para a página inicial de upload de arquivo
@app.route('/', methods=['GET', 'POST'])
def upload():
    form = UploadForm()  # Instância do formulário de upload

    # Verifica se o formulário foi submetido corretamente
    if form.validate_on_submit():
        file = form.file.data  # Obtém o arquivo enviado

        # Verifica se o arquivo é permitido
        if file and allowed_file(file.filename):
            filename = 'car_data.csv'  # Nome fixo para o arquivo
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # Salva o arquivo no diretório configurado

            flash('Upload realizado com sucesso!')  # Mensagem de sucesso
            return redirect(url_for('analysis'))  # Redireciona para a página de análise

        else:
            flash('Arquivo inválido. Apenas arquivos CSV são permitidos.')  # Mensagem de erro para tipos inválidos

    # Renderiza a página de upload com o formulário
    return render_template('upload.html', form=form)

# Rota para a página de análise de dados
@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'car_data.csv')  # Caminho completo do arquivo de dados

    try:
        df = pd.read_csv(filepath)  # Lê o arquivo CSV em um DataFrame
    except FileNotFoundError:
        flash('Nenhum arquivo carregado. Faça upload dos dados.')  # Mensagem de erro se o arquivo não existir
        return redirect(url_for('upload'))  # Redireciona para a página de upload

    form = ModelForm()  # Instância do formulário para treinamento do modelo
    metrics = None  # Inicializa as métricas como vazias
    model_trained = False  # Indica se o modelo foi treinado

    # Gera gráficos interativos para análise exploratória
    brand_chart, year_chart = generate_interactive_plots(df)

    # Prepara o dataset para o modelo
    df, scaler = prepare_dataset(df)
    X = df.drop('selling_price', axis=1)  # Separação das features
    y = df['selling_price']  # Separação da variável alvo

    # Verifica se o formulário foi submetido corretamente
    if form.validate_on_submit():
        model_type = form.model_type.data  # Obtém o tipo de modelo escolhido
        param = form.param.data  # Obtém o parâmetro do modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Divide os dados

        # Escolhe o modelo baseado no tipo selecionado
        if model_type == 'DecisionTree':
            model = DecisionTreeRegressor(max_depth=param, random_state=42)
        elif model_type == 'RandomForest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=param, random_state=42)
        elif model_type == 'GradientBoosting':
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=param, random_state=42)
        else:
            flash('Tipo de modelo inválido.')  # Mensagem de erro para tipo inválido
            return redirect(request.url)

        model.fit(X_train, y_train)  # Treina o modelo com os dados de treinamento
        y_pred = model.predict(X_test)  # Faz predições no conjunto de teste

        # Calcula as métricas do modelo
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        metrics = {
            'R2': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'CV R2 Mean': cv_scores.mean()
        }

        save_model(model, X.columns.tolist(), scaler)  # Salva o modelo treinado e os metadados
        model_trained = True  # Atualiza o status de treinamento

    # Renderiza a página de análise com os resultados e gráficos
    return render_template('analysis.html', form=form, metrics=metrics, model_trained=model_trained,
                           brand_chart=brand_chart, year_chart=year_chart)

# Rota para a página de previsão de preços
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = PredictionForm()  # Instância do formulário de previsão

    try:
        model, model_columns, scaler = load_model()  # Carrega o modelo salvo
    except FileNotFoundError:
        flash('Modelo não encontrado. Por favor, treine o modelo primeiro.')  # Mensagem de erro se o modelo não existir
        return redirect(url_for('analysis'))  # Redireciona para a página de análise

    # Define as opções de marca no formulário
    brand_columns = [col for col in model_columns if col.startswith('brand_')]
    brands = [col.replace('brand_', '') for col in brand_columns]
    form.brand.choices = [(brand, brand) for brand in brands]

    prediction = None  # Inicializa a variável de previsão como vazia

    # Verifica se o formulário foi submetido corretamente
    if form.validate_on_submit():
        input_dict = {col: 0 for col in model_columns}  # Inicializa o dicionário de entrada com zeros

        # Define os valores para as características categóricas
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

        # Processa as características numéricas
        age = form.age.data
        km_driven = form.km_driven.data
        numerical_features = pd.DataFrame({'age': [age], 'km_driven': [np.log1p(km_driven)]})
        numerical_features_scaled = scaler.transform(numerical_features)

        input_dict['age'] = numerical_features_scaled[0][0]
        input_dict['km_driven'] = numerical_features_scaled[0][1]

        input_df = pd.DataFrame([input_dict], columns=model_columns)  # Cria o DataFrame de entrada

        log_prediction = model.predict(input_df)  # Faz a previsão
        prediction = np.expm1(log_prediction[0])  # Converte a previsão logarítmica para o valor original

    # Renderiza a página de previsão com o formulário e o resultado
    return render_template('predict.html', form=form, prediction=prediction)

# Inicia a aplicação Flask
if __name__ == '__main__':
    app.run(debug=True)
