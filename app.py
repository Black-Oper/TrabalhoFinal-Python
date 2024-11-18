from flask import Flask, request, render_template, redirect, url_for, flash
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import config
from config import allowed_file
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from modelForm import ModelForm

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
            if not set(expected_columns).issubset(df.columns):
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
    map_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    form = ModelForm()
    model_trained = False
    score = None

    if request.method == 'POST' and form.validate_on_submit():
        # Remover linhas nulas
        df = df.dropna()

        df['marca'] = df['marca'].astype(str)
        df['modelo'] = df['modelo'].astype(str)
        df['cidade'] = df['cidade'].astype(str)
        df['quilometragem'] = df['quilometragem'].astype(int)
        df['estado'] = df['estado'].astype(str)
        df['preco'] = df['preco'].astype(float)
        df['ano'] = df['ano'].astype(int)

        X = df[['marca', 'modelo', 'ano', 'quilometragem']]
        y = df['preco']

        X = pd.get_dummies(X, columns=['marca', 'modelo'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param = form.param.data

        if form.model_type.data == 'Decision Tree':
            model = DecisionTreeRegressor(max_depth=param, random_state=42)
        elif form.model_type.data == 'KNN':
            model = KNeighborsRegressor(n_neighbors=param)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        score = r2_score(y_test, y_pred)
        print(f'R²: {score}')

        model_trained = True
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
        score=score,
        map_html=map_html
    )


# Início do servidor
if __name__ == '__main__':
    app.run(debug=True)
