from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, IntegerField
from wtforms.validators import DataRequired, NumberRange
from flask_wtf.file import FileField, FileRequired

class UploadForm(FlaskForm):
    file = FileField('Arquivo', validators=[FileRequired()])
    submit = SubmitField('Fazer Upload')

class ModelForm(FlaskForm):
    model_type = SelectField('Tipo de Modelo', choices=[
        ('DecisionTree', 'Decision Tree'),
        ('RandomForest', 'Random Forest'),
        ('GradientBoosting', 'Gradient Boosting')
    ], validators=[DataRequired()])
    param = IntegerField('Parâmetro', validators=[DataRequired()])
    submit = SubmitField('Treinar Modelo')

class PredictionForm(FlaskForm):
    brand = SelectField('Marca', choices=[], validators=[DataRequired()])
    age = IntegerField('Idade do Veículo (anos)', validators=[DataRequired(), NumberRange(min=0)])
    km_driven = IntegerField('Quilômetros Rodados', validators=[DataRequired(), NumberRange(min=0)])
    fuel = SelectField('Combustível', choices=[
        ('Petrol', 'Gasolina'),
        ('Diesel', 'Diesel'),
        ('CNG', 'GNV'),
        ('LPG', 'GLP'),
        ('Electric', 'Elétrico')
    ], validators=[DataRequired()])
    transmission = SelectField('Transmissão', choices=[
        ('Manual', 'Manual'),
        ('Automatic', 'Automática')
    ], validators=[DataRequired()])
    seller_type = SelectField('Tipo de Vendedor', choices=[
        ('Dealer', 'Revendedor'),
        ('Individual', 'Particular'),
        ('Trustmark Dealer', 'Revendedor Certificado')
    ], validators=[DataRequired()])
    owner = SelectField('Número de Proprietários', choices=[
        ('First Owner', 'Primeiro Dono'),
        ('Second Owner', 'Segundo Dono'),
        ('Third Owner', 'Terceiro Dono'),
        ('Fourth & Above Owner', 'Quarto ou Mais'),
        ('Test Drive Car', 'Carro de Teste')
    ], validators=[DataRequired()])
    submit = SubmitField('Prever Preço de Venda')
    