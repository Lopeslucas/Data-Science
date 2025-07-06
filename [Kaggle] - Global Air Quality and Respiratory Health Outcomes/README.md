# Global Air Quality and Respiratory Health Outcomes - Data Science Study

Este repositório contém um notebook de estudo e experimentação em Ciência de Dados, utilizando o conjunto de dados global de qualidade do ar e internações hospitalares por problemas respiratórios.

## Estrutura do Projeto

- **[Kaggle] - Global Air Quality and Respiratory Health Outcomes/**
  - `EDA-v1.ipynb`: Análise exploratória, tratamento de dados, engenharia de features e modelagem preditiva.
  - `air_quality_health_dataset.csv`: Base de dados utilizada (não versionar se for grande/confidencial).

## Objetivos
- Explorar e entender o conjunto de dados.
- Realizar pré-processamento e análise exploratória.
- Construir e avaliar modelos de regressão para prever internações hospitalares.
- Praticar técnicas de engenharia de features e avaliação de modelos.

## Técnicas Utilizadas
- **Leitura e análise exploratória:**
    - Carregamento do dataset com pandas.
    - Análise de tipos, valores nulos, duplicados e estatísticas descritivas.
    - Verificação de valores únicos em colunas categóricas.
- **Visualização de dados:**
    - Histogramas e boxplots para distribuição e outliers.
    - Heatmap de correlação (Pearson e Spearman).
    - Análise temporal de hospitalizações.
    - Boxplot por categoria (population_density).
- **Pré-processamento:**
    - Conversão de datas e extração de ano, mês, dia e dia da semana.
    - Separação correta de treino e teste antes do tratamento.
    - Teste de normalidade (D’Agostino) nas variáveis numéricas.
    - Tratamento de outliers apenas no conjunto de treino (IQR).
- **Modelagem preditiva:**
    - RandomForestRegressor e GradientBoostingRegressor.
    - Avaliação com Mean Squared Error (MSE) e R² Score.
    - Visualização da importância das variáveis.

## Próximos Passos para Melhoria
- **Feature Engineering:** Criar novas variáveis derivadas, explorar interações, médias móveis, lags temporais, etc.
- **Variáveis Categóricas:** Aplicar encoding adequado (ex: OneHotEncoder) para ‘city’ e ‘population_density’.
- **Escalonamento:** Aplicar scaler nas variáveis numéricas para beneficiar alguns modelos.
- **Validação Cruzada:** Utilizar cross-validation para avaliação mais robusta dos modelos.
- **Modelos:** Testar outros algoritmos (ex: regressão regularizada, XGBoost, redes neurais).
- **Análise Temporal:** Mesmo com datas sintéticas, tentar agrupamentos por mês/ano para suavizar ruídos.
- **Explicabilidade:** Analisar resíduos, gráficos de previsão vs. real, e utilizar ferramentas como SHAP/LIME.
- **Dados:** Buscar mais variáveis externas (clima, eventos, políticas públicas) para enriquecer o modelo.
- **Documentação:** Adicionar mais comentários explicando decisões e interpretações dos resultados.


## Referências

- [Kaggle Global Air Quality and Respiratory Health Outcomes](https://www.kaggle.com/datasets/tfisthis/global-air-quality-and-respiratory-health-outcomes)

---

*Projeto de estudos em Ciência de Dados - Lucas Lopes*