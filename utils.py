import os  # Módulo para operações no sistema de arquivos
from flask import Flask  # Framework web Flask
import joblib  # Biblioteca para salvar e carregar modelos treinados
import numpy as np  # Biblioteca para operações numéricas
import pandas as pd  # Biblioteca para manipulação de dados
from sklearn.preprocessing import StandardScaler  # Para normalizar os dados
import plotly.express as px  # Para criar gráficos interativos

# Inicializa a aplicação Flask
app = Flask(__name__)
app.secret_key = 'mister-picanha'  # Chave secreta usada pelo Flask para segurança (ex.: CSRF)

# Função para verificar se um arquivo tem a extensão permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Função para carregar um modelo treinado
def load_model():
    model_path = os.path.join(app.config['MODEL_FOLDER'], 'model.pkl')  # Caminho completo para o arquivo do modelo
    data = joblib.load(model_path)  # Carrega o arquivo usando joblib
    return data['model'], data['columns'], data['scaler']  # Retorna o modelo, as colunas e o scaler

# Função para salvar um modelo treinado
def save_model(model, columns, scaler):
    model_path = os.path.join(app.config['MODEL_FOLDER'], 'model.pkl')  # Caminho completo para salvar o modelo
    joblib.dump({'model': model, 'columns': columns, 'scaler': scaler}, model_path)  # Salva o modelo e os metadados

# Função para preparar o dataset para treinamento e previsão
def prepare_dataset(df):
    df = df.dropna()  # Remove linhas com valores nulos

    # Extrai a marca do veículo da coluna 'name'
    df['brand'] = df['name'].str.split().str[0]

    # Calcula a idade do veículo
    df['age'] = 2024 - df['year']

    # Aplica transformação logarítmica para reduzir variabilidade em 'km_driven' e 'selling_price'
    df['km_driven'] = np.log1p(df['km_driven'])
    df['selling_price'] = np.log1p(df['selling_price'])

    # Remove a coluna 'name', pois não será utilizada no modelo
    df = df.drop('name', axis=1)

    # Cria variáveis dummy para colunas categóricas
    df = pd.get_dummies(df, columns=['brand', 'fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

    # Normaliza os valores das colunas 'age' e 'km_driven'
    scaler = StandardScaler()
    df[['age', 'km_driven']] = scaler.fit_transform(df[['age', 'km_driven']])

    return df, scaler  # Retorna o DataFrame preparado e o scaler usado

# Função para gerar gráficos interativos
def generate_interactive_plots(df):
    # Gráfico de barras: distribuição por marca
    brand_counts = df['name'].str.split().str[0].value_counts().reset_index()  # Conta as ocorrências de cada marca
    brand_counts.columns = ['Brand', 'Count']  # Renomeia as colunas
    fig1 = px.bar(brand_counts, x='Brand', y='Count', title='Distribuição por Marca')  # Cria o gráfico interativo
    brand_chart = fig1.to_html(full_html=False)  # Converte o gráfico para HTML (parcial)

    # Gráfico de barras: distribuição por ano
    year_counts = df['year'].value_counts().reset_index()  # Conta as ocorrências de cada ano
    year_counts.columns = ['Year', 'Count']  # Renomeia as colunas
    fig2 = px.bar(year_counts, x='Year', y='Count', title='Distribuição por Ano')  # Cria o gráfico interativo
    year_chart = fig2.to_html(full_html=False)  # Converte o gráfico para HTML (parcial)

    return brand_chart, year_chart  # Retorna os gráficos em formato HTML
