from flask_wtf import FlaskForm  # Importa a classe base para formulários do Flask-WTF
from wtforms import SelectField, SubmitField, IntegerField  # Importa os campos de formulário
from wtforms.validators import DataRequired, NumberRange  # Importa os validadores de formulário
from flask_wtf.file import FileField, FileRequired  # Importa campos específicos para upload de arquivos

# Classe para o formulário de upload de arquivo
class UploadForm(FlaskForm):
    # Campo para selecionar o arquivo (obrigatório)
    file = FileField('Arquivo', validators=[FileRequired()])
    # Botão para submeter o formulário
    submit = SubmitField('Fazer Upload')

# Classe para o formulário de treinamento do modelo
class ModelForm(FlaskForm):
    # Campo para selecionar o tipo de modelo a ser treinado
    model_type = SelectField('Tipo de Modelo', choices=[
        ('DecisionTree', 'Decision Tree'),  # Opção: Árvore de Decisão
        ('RandomForest', 'Random Forest'),  # Opção: Floresta Aleatória
        ('GradientBoosting', 'Gradient Boosting')  # Opção: Boosting Gradiente
    ], validators=[DataRequired()])  # Validador para garantir que o campo seja preenchido

    # Campo para inserir o parâmetro do modelo
    param = IntegerField('Parâmetro', validators=[DataRequired()])
    # Botão para submeter o formulário
    submit = SubmitField('Treinar Modelo')

# Classe para o formulário de previsão de preço
class PredictionForm(FlaskForm):
    # Campo para selecionar a marca do veículo (carregado dinamicamente)
    brand = SelectField('Marca', choices=[], validators=[DataRequired()])
    # Campo para inserir a idade do veículo (com validação para valores positivos)
    age = IntegerField('Idade do Veículo (anos)', validators=[DataRequired(), NumberRange(min=0)])
    # Campo para inserir a quilometragem rodada (com validação para valores positivos)
    km_driven = IntegerField('Quilômetros Rodados', validators=[DataRequired(), NumberRange(min=0)])
    # Campo para selecionar o tipo de combustível
    fuel = SelectField('Combustível', choices=[
        ('Petrol', 'Gasolina'),
        ('Diesel', 'Diesel'),
        ('CNG', 'GNV'),
        ('LPG', 'GLP'),
        ('Electric', 'Elétrico')
    ], validators=[DataRequired()])
    # Campo para selecionar o tipo de transmissão
    transmission = SelectField('Transmissão', choices=[
        ('Manual', 'Manual'),
        ('Automatic', 'Automática')
    ], validators=[DataRequired()])
    # Campo para selecionar o tipo de vendedor
    seller_type = SelectField('Tipo de Vendedor', choices=[
        ('Dealer', 'Revendedor'),
        ('Individual', 'Particular'),
        ('Trustmark Dealer', 'Revendedor Certificado')
    ], validators=[DataRequired()])
    # Campo para selecionar o número de proprietários anteriores
    owner = SelectField('Número de Proprietários', choices=[
        ('First Owner', 'Primeiro Dono'),
        ('Second Owner', 'Segundo Dono'),
        ('Third Owner', 'Terceiro Dono'),
        ('Fourth & Above Owner', 'Quarto ou Mais'),
        ('Test Drive Car', 'Carro de Teste')
    ], validators=[DataRequired()])
    # Botão para submeter o formulário
    submit = SubmitField('Prever Preço de Venda')
