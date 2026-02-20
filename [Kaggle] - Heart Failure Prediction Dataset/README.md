# Heart Failure Prediction Dataset
Esta pasta contém um notebook de estudo e experimentação em Ciência de Dados, utilizando o conjunto de dados de doença cardiaca disponível na Kaggle.

## Objetivos
- Explorar e entender o conjunto de dados.
- Realizar pré-processamento e análise exploratória.
- Construir e avaliar modelos de classificação para prever doença cardiaca
- Praticar técnicas de otimização de modelos.

## Estrutura do Notebook
### 1 - Importaçaão das Bibliotecas e Leitura Inicial da Base
- Leitura do Head
- Tamanho do DF
- Informações complementares (info)
- Descrição/ distribuição dos dados

### 2 - Analise Grafica
- Divisão do DF, entre as variaveis numericas e categoricas
- Boxplot 
- Violinplot
- Histplot 

### 3 - Divisão dos Dados e Implementação de Modelo
- Divisão das variaveis explicaticas da variavel alvo
- Divisão dos Dados de Treino e Test
- Divisão das variaveis numericas e categoricas
- Pipeline de Preprocessamento dos Dados:
	- Tratamento das variaveis categoricas com ***OneHotEncoder*** (Sex, ChestPainType, RestingECG, ST_Slope e ExerciseAngina)
	- Tratamento das variaveis numericas com ***padronização/ StandardScaler*** (Age, RestingBP, Cholesterol, FastingBS, MaxHR e Oldpeak).

***Notas:***
- Por que fazer a divisão antes do tratamento dos dados?
Porque qualquer transformação que utilize estatísticas globais pode introduzir data leakage, tornando o modelo otimista artificialmente e prejudicando a generalização.	

- Implementação do Modelo 

### Metricas de Avaliação
- AUC: 93%
- Matrix de Confusão:
	- TN = 68
	- FP = 14
	- FN = 7
	- TP = 95
***Notas:*** 
- 14 falsos positivos: Que podem ocasionar em impacto de exames desnecessarios. Alerta para pessoas saudaveis que podem trazer um impacto. Porem no contexto de Saude, é um trade-off justo, onde pode se interpretar melhor um falso positivo do que um falso negativo (onde a pessoa vai ter sido informada estar saudavel, porem esta com algum problema de saude).
- 7 falsos negativos: Como reduzir ainda mais os falsos negativos?
Ajustando o threshold da regressão logística para aumentar o recall da classe positiva, priorizando sensibilidade em vez de precisão. Em saúde, deve se priorizar a redução de **falsos negativos**

### 4 - Validação Cruzada
Implementação da validação cruzada para garantir que a minha metrica de AUC em 93% não foi sorte ocasionada pela divisão (split), garantir que eu tenho um modelo estavel.
- AUC por Fold: [0.94081779 0.93938307 0.91846007 0.89471142 0.85124366]
- AUC Médio 0.9089232022197915
- Desvio Padrão 0.0333614276783346
***Notas:***
Tivemos uma boa media, porem tivemos um dos folds com uma AUC muito baixa em relação as demais. Indicando possiveis outliers e confirmando que os meus dados não possuem uma distribuição linear (oque ja havia sido confirmado na sessão de analises graficas). Devido a isso, testar algum outro modelo que não dependa da Linearidade dos Dados.

### 5 - Implementação do modelo Random Forrest
- Utilizando os Hiperparametros: 
	n_estimators = 300 árvores
	max_depth = None, Árvores sem limite de profundidade
	random_state = 42, Resultado reprodutível
	n_jobs = -1, Paralelização máxima da CPU

#### 5.1 - Validação cruzada - Random Forrest
- AUC por Fold: [0.95139885 0.93747011 0.91266141 0.9033446  0.84442164]
- AUC Médio 0.9098593216566992
- Desvio Padrão 0.03693310711938147
***Notas:***
O modelo de Regressão Logistica, esta com uma estabilidade melhor que o RandomForrest





## Referências

- [Kaggle Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data)

---

*Projeto de estudos em Ciência de Dados