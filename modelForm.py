from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired


class ModelForm(FlaskForm):
    model_type = SelectField('Escolha o Modelo', choices=[('Decision Tree', 'Árvore de Decisão'), ('KNN', 'KNN')], validators=[DataRequired()])
    param = IntegerField('Parâmetro (Profundidade da Árvore ou Nº de Vizinhos)', validators=[DataRequired()])
    submit = SubmitField('Treinar Modelo')