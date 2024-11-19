from flask import Flask, request, render_template, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, IntegerField
from wtforms.validators import DataRequired
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Configuração do Flask
app = Flask(__name__)
app.secret_key = 'mister-picanha'

# Configurações de diretórios
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}
app.config['STATIC_FOLDER'] = 'static'

# Garantir a criação das pastas necessárias
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Função auxiliar para verificar arquivos permitidos
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Definição do formulário
class ModelForm(FlaskForm):
    model_type = SelectField('Selecione o Modelo', choices=[
        ('Decision Tree', 'Decision Tree'),
        ('Random Forest', 'Random Forest')
    ])
    param = IntegerField('Parâmetro', validators=[DataRequired()])
    submit = SubmitField('Treinar Modelo')

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
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                df = pd.read_csv(filepath)
                expected_columns = ['marca', 'modelo', 'ano', 'preco', 'quilometragem', 'cidade', 'estado', 'latitude', 'longitude']
                if not set(expected_columns).issubset(df.columns):
                    flash('As colunas do arquivo não correspondem ao esperado.')
                    return redirect(request.url)
            except Exception as e:
                flash(f'Erro ao processar arquivo: {str(e)}')
                return redirect(request.url)

            return redirect(url_for('analysis', filename=filename))
        else:
            flash('Arquivo inválido. Apenas arquivos CSV são permitidos.')

    return render_template('upload.html')

# Página de análise
@app.route('/analysis/<filename>', methods=['GET', 'POST'])
def analysis(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        df = pd.read_csv(filepath)
        df.dropna(inplace=True)

        df['marca'] = df['marca'].astype(str)
        df['modelo'] = df['modelo'].astype(str)
        df['cidade'] = df['cidade'].astype(str)
        df['quilometragem'] = pd.to_numeric(df['quilometragem'], errors='coerce')
        df['estado'] = df['estado'].astype(str)
        df['preco'] = pd.to_numeric(df['preco'], errors='coerce')
        df['ano'] = pd.to_numeric(df['ano'], errors='coerce')
    except Exception as e:
        flash(f'Erro ao processar os dados: {str(e)}')
        return redirect(url_for('upload_file'))

    scaler = StandardScaler()
    df['quilometragem'] = scaler.fit_transform(df[['quilometragem']])

    # Gráficos e análise
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='marca', order=df['marca'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title('Distribuição por Marca')
    bar_chart_path1 = os.path.join(app.config['STATIC_FOLDER'], 'bar_chart1.png')
    plt.savefig(bar_chart_path1)
    plt.close()

    # Segundo gráfico
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='estado', order=df['estado'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title('Distribuição por Estado')
    bar_chart_path2 = os.path.join(app.config['STATIC_FOLDER'], 'bar_chart2.png')
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
    map_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    form = ModelForm()
    model_trained = False
    score = None

    if request.method == 'POST' and form.validate_on_submit():
        X = df[['marca', 'modelo', 'ano', 'quilometragem']]
        y = df['preco']
        X = pd.get_dummies(X, columns=['marca', 'modelo'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_value = form.param.data

        if form.model_type.data == 'Decision Tree':
            model = DecisionTreeRegressor(max_depth=param_value, random_state=42)
            model.fit(X_train, y_train)
        elif form.model_type.data == 'Random Forest':
            model = RandomForestRegressor(n_estimators=param_value, random_state=42)
            model.fit(X_train, y_train)
        else:
            flash('Modelo inválido selecionado.')
            return redirect(url_for('analysis', filename=filename))

        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        model_trained = True

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

if __name__ == "__main__":
    app.run(debug=True)
