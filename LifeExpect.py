import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

# TRATAMENTO DOS DADOS
# Função para renomear as colunas
def rename_columns(col):
    return col.strip().replace(' ', '_').lower()


# Existem celulas em branco, iremos preencher essas celulas em brancos com NaN 
def tratamento_dados(df):
    # Aplicar a função em todos os nomes das colunas
    df.columns = [rename_columns(col) for col in df.columns]
    print(df.columns) 

    df = df.replace(" ", np.NaN)

    df['life_expectancy'] = df['life_expectancy'].fillna(df['life_expectancy'].mean())
    df['adult_mortality'] = df['adult_mortality'].fillna(df['adult_mortality'].mean())
    df['alcohol'] = df['alcohol'].fillna(df['alcohol'].mean())
    df['hepatitis_b'] = df['hepatitis_b'].fillna(df['hepatitis_b'].mean())
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
    df['polio'] = df['polio'].fillna(df['polio'].mean())
    df['total_expenditure'] = df['total_expenditure'].fillna(df['total_expenditure'].mean())
    df['diphtheria'] = df['diphtheria'].fillna(df['diphtheria'].mean())
    df['gdp'] = df['gdp'].fillna(df['gdp'].mean())
    df['population'] = df['population'].fillna(df['population'].mean())
    df['thinness__1-19_years'] = df['thinness__1-19_years'].fillna(df['thinness__1-19_years'].mean())
    df['thinness_5-9_years'] = df['thinness_5-9_years'].fillna(df['thinness_5-9_years'].mean())
    df['income_composition_of_resources'] = df['income_composition_of_resources'].fillna(df['income_composition_of_resources'].mean())
    df['schooling'] = df['schooling'].fillna(df['schooling'].mean())

    # Criar uma instância de LabelEncoder
    label_encoder = LabelEncoder()

    df['country'] = label_encoder.fit_transform(df['country']).astype(int) # transformação de string em int
    df['status'] = label_encoder.fit_transform(df['status']).astype(int) # transformação de string em int

    return df

# REGRESSÃO LINEAR 

def regressao_linear(df):
    # TRAIN TEST SPLIT 
    # Para a regressão as variáveis serão x (como um conjunto de todas as colunas menos o recurso alvo a ser observado) e Y (life_expectancy)
    X = df.drop(['life_expectancy'], axis =1)
    y = df['life_expectancy']

    # . Dividir os dados em conjuntos de treinamento e teste permite avaliar o desempenho do modelo em dados não vistos, ajudando a identificar se o 
    # modelo está superestimando (overfitting) ou subestimando (underfitting) os dados. No nosso caso, 30% dos dados serão utilizados para teste e 70% para treinamento.

    X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    # Criando uma instância MinMaxScaler (método de normalização que transforma os dados de um intervalo específico, geralmente entre 0 e 1.)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_org)
    X_test = scaler.transform(X_test_org)

    print("X_train shape: ",X_train.shape)
    print("y_train shape: ",y_train.shape)
    print("X_test shape: ",X_test.shape)
    print("y_test shape",y_test.shape)

    # Em resultado vimos que 2056 são X_train e 882 são X_test 

    # Criando uma Instância do Modelo de Regressão Linear
    lreg = LinearRegression()
    # Ajusta o modelo de regressão linear (lreg) aos dados de treinamento normalizados (X_train e y_train)
    lreg.fit(X_train, y_train)

    print('Train Score: {:.4f}'.format(lreg.score(X_train, y_train)))
    print('Test Score:{:.4f}'.format(lreg.score(X_test, y_test)))

    ylinear_predicted = lreg.predict(X_test)

    print('MSE:', metrics.mean_squared_error(y_test,ylinear_predicted))
    print('R2_score: {:.4f}'.format(metrics.r2_score(y_test,ylinear_predicted)))

    # Plotando os valores reais versus os valores previstos pelo modelo
    plt.scatter(ylinear_predicted, y_test, color = 'blue')
    plt.plot(y_test, y_test, color='red', label='Linha de Regressão')

    # Adicionando legenda
    plt.legend()

    plt.show()

# FUNÇÕES PARA ANÁLISE

# Matriz com subconjuntos de todas as colunas

def grafico_matriz_colunas(df):
    cols = df.columns[4:9]
    y = df['life_expectancy']
    scatter_matrix(df[cols], figsize = (20,20), c = y, alpha = 0.8, marker = 'O')

    plt.show()
    

# Matrix de correlação (Matriz de correlação mostra as relações entre todas as colunas numéricas do DataFrame)

def grafico_matriz_correlacao(df):
    # Cria os eixos para o Gráfico
    corr = df.corr() 
    sns.heatmap(corr, annot = True) 

    plt.show()

def grafico_distribuicao_country_life_expectancy(df):
    # Agrupar os Dados por 'country' e Calcular a Média de 'life_expectancy'
    # Considerando que os todos dados coletados são de 2000 a 2015, vamos agrupar a informação apenas por país
    dfpais = df.groupby('country')['life_expectancy'].mean().reset_index()

    # Gráfico de distribuição (gráfico de distribuição para a coluna 'life_expectancy' no DataFrame dfpais, adicionando uma curva KDE para mostrar a estimativa da densidade de probabilidade)
    sns.displot(dfpais['life_expectancy'],kde=True)

    plt.show()

def grafico_pairplot(df):
    # Pairplot (pairplot mostra distribuições univariadas e relações bivariadas entre diferentes colunas do DataFrame)
    sns.set_theme()
    # Selecionando apenas algumas colunas
    cols_to_plot = ['life_expectancy', 
                    'adult_mortality', 
                    'infant_deaths', 
                    'alcohol', 
                    'hepatitis_b', 
                    'polio',
                    'hiv/aids',
                    'population']

    df_selected = df[cols_to_plot]

    sns.pairplot(df_selected) 

    plt.show()

#----------------------------------------------------

# Ler o arquivo CSV
df = pd.read_csv('bd/LifeExpectancyData.csv')


# INFORMAÇÕES DA DATAFRAME

# Verifica se df tem valores Null

isnull = df.isnull().sum()
#print(isnull)

# Aplica a função de tratamento de dados
df =  tratamento_dados(df)

# Mostra as informações de cabeçalho
cabecalho = df.head()
#print(cabecalho) 

# Mostra os tipos de Colunas

tipo_colunas =  df.dtypes
#print(tipo_colunas)

# Mostra a informação sobre Non-Null, datatype.. 

informacao =  df.info()
#print(informacao)


# Porcentagem de valores Null 
porcentagem_null = df.isna().mean().round(4) * 100
#print(porcentagem_null)

nova_porcentagem_null = df.isna().sum() 
#print(nova_porcentagem_null)

descricao =  df.describe()
#print(descricao)


# Gera gráficos

grafico_pairplot(df)
grafico_distribuicao_country_life_expectancy(df)
grafico_matriz_correlacao(df)
grafico_matriz_colunas(df)
regressao_linear(df)

