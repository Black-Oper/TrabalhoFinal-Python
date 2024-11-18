from flask import Flask, request, render_template, redirect, url_for, flash
from sklearn.metrics import accuracy_score, mean_squared_error
import config
from config import allowed_file, prepare_data, train_model
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from modelForm import ModelForm
import numpy as np

# Configuração do Flask
app = Flask(__name__)
app.secret_key = 'mister-picanha'

# Configurações de diretórios
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = config.MODEL_FOLDER
app.config['ALLOWED_EXTENSIONS'] = config.ALLOWED_EXTENSIONS

# Página de upload
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Nenhum arquivo enviado.')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('Nenhum arquivo selecionado.')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config.get('UPLOAD_FOLDER', './uploads'), filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            # Validação das colunas
            expected_columns = ['marca', 'modelo', 'ano', 'preco', 'quilometragem', 'cidade', 'estado', 'latitude', 'longitude']
            if list(df.columns) != expected_columns:
                flash('As colunas do arquivo não correspondem ao esperado.')
                return redirect(request.url)

            return redirect(url_for('analysis', filename=filename))
        else:
            flash('Arquivo inválido. Apenas arquivos CSV são permitidos.')

    return render_template('upload.html')

# Página de análise
@app.route('/analysis/<filename>', methods=['GET', 'POST'])
def analysis(filename):
    filepath = os.path.join(app.config.get('UPLOAD_FOLDER', './uploads'), filename)
    df = pd.read_csv(filepath)

    # Gráfico 1: Distribuição por Marca
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='marca', order=df['marca'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title('Distribuição por Marca')
    bar_chart_path1 = os.path.join('static', 'bar_chart1.png')
    plt.savefig(bar_chart_path1)
    plt.close()

    # Gráfico 2: Distribuição por Estado
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='estado', order=df['estado'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title('Distribuição por Estado')
    bar_chart_path2 = os.path.join('static', 'bar_chart2.png')
    plt.savefig(bar_chart_path2)
    plt.close()

    # Mapa interativo com Plotly
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        hover_name="modelo",
        color="preco",
        zoom=4,
        height=500
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    map_html = os.path.join('templates', 'map.html')
    fig.write_html(map_html, full_html=False, include_plotlyjs='cdn')

    # Formulário e treinamento do modelo
    form = ModelForm()
    model_trained = False
    score = {}

    if request.method == 'POST' and form.validate_on_submit():
        X_train, X_test, y_train, y_test = prepare_data(df)
        params = {}

        if form.model_type.data == 'Decision Tree':
            params['max_depth'] = form.param.data
        elif form.model_type.data == 'KNN':
            params['n_neighbors'] = form.param.data

        model = train_model(X_train, y_train, form.model_type.data, params)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        model_trained = True
        score = {'accuracy': acc, 'rmse': rmse}
    else:
        if form.errors:
            flash(form.errors)

    return render_template(
        'analysis.html',
        bar_chart_url1='bar_chart1.png',
        bar_chart_url2='bar_chart2.png',
        form=form,
        filename=filename,
        model_trained=model_trained,
        score=score
    )

# Início do servidor
if __name__ == '__main__':
    app.run(debug=True)
