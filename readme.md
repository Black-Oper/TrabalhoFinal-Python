# Projeto de Previsão de Preço de Carros

## Visão Geral
Este projeto tem como objetivo prever o preço de venda de carros usados com base em várias características, como modelo, ano, quilometragem, tipo de combustível, tipo de vendedor, transmissão e número de donos anteriores. O projeto utiliza modelos de machine learning para fazer as previsões e fornece uma interface web para que os usuários possam carregar dados de carros e obter previsões de preço.

## Funcionalidades
- Upload e pré-processamento de dados
- Treinamento e avaliação do modelo de machine learning
- Interface web para upload de dados e previsão de preço
- Análise de dados e visualizações

## Bibliotecas Usadas
- os
- pandas
- numpy
- matplotlib
- flask
- flask_wtf
- wtforms
- sklearn
- plotly

## Estrutura de Arquivos
- `app.py`: Arquivo principal da aplicação para rodar o servidor Flask.
- `Forms.py`: Contém a definição dos formulários para upload de arquivos e inputs do usuário.
- `utils.py`: Funções auxiliares para pré-processamento de dados e treinamento do modelo.
- `templates/`: Contém os templates HTML para a interface web.
  - `upload.html`: Template para upload dos dados de carros.
  - `predict.html`: Template para exibição das previsões de preço.
  - `analysis.html`: Template para análise de dados e visualizações.
- `static/`: Contém arquivos estáticos.
  - `/css`: Contém os estilos CSS para a interface web.
  - `/img`: Contém imagens utilizadas na interface web.
- `uploads/`: Diretório para armazenar os arquivos de dados carregados.
  - `car_data.csv`: Arquivo de dados de carros de exemplo.

## Como Rodar
1. Clone o repositório.
2. Instale as bibliotecas necessárias.
3. Rode a aplicação usando `flask run`.
4. Abra um navegador e acesse `http://localhost:5000` (verifique a porta) para acessar a interface web.

## Uso
1. Faça o upload de um arquivo CSV contendo os dados dos carros usando o formulário de upload.
2. A aplicação irá pré-processar os dados e treinar os modelos de machine learning.
3. Utilize o formulário de previsão para inserir as características de um carro e obter o preço de venda previsto.
4. Visualize as análises de dados e visualizações na página de análise.

## Conjunto de Dados
O conjunto de dados contém informações sobre vários carros, incluindo:
- `name`: Nome do modelo do carro
- `year`: Ano de fabricação do carro
- `selling_price`: Preço de venda do carro
- `km_driven`: Quilometragem percorrida pelo carro
- `fuel`: Tipo de combustível utilizado pelo carro (Petrol, Diesel, CNG, LPG, Elétrico)
- `seller_type`: Tipo de vendedor (Individual, Dealer, Trustmark Dealer)
- `transmission`: Tipo de transmissão (Manual, Automática)
- `owner`: Número de donos anteriores (Primeiro Dono, Segundo Dono, etc.)

## Licença
Este projeto está licenciado sob a Licença MIT.
