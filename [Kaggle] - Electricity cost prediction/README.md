# Electricity Cost Prediction Dataset

Este conjunto de dados, intitulado "Fortalecendo Previsões: Revelando a Dinâmica dos Custos de Eletricidade na Infraestrutura Moderna", fornece uma coleção realista e abrangente de 10.000 pontos de dados projetados para previsão de custos de eletricidade. Ele abrange uma variedade de fatores que influenciam o consumo e as despesas de energia em diferentes tipos de locais.

## Estrutura do Projeto
**[Kaggle] - Electricity Cost Prediction Dataset/**
- `EDA.ipynb`: Análise exploratória, tratamento de dados, seleção de features, modelos de Regressão e comparação de métricas.
- `electricity_cost_dataset.csv`: Base de dados utilizada

## Objetivos
- ✅ Explorar e entender o conjunto de dados.
- ✅ Realizar pré-processamento e análise exploratória.
- ✅ Construir e avaliar modelos de regressão para prever o valor pago na "conta de energia".
- ✅ Comparar técnicas e métricas de avaliação.
- ⏳ Conclusão para "Negócio"

## Técnicas Utilizadas
- Análise Exploratória de Dados (EDA)
- Visualização de Dados
    - Boxplot para identificar distribuição e outliers nas variáveis numéricas
    - Boxplot para analisar a variável categórica (structure type) versus a variável alvo
    - Histplot para visualizar a distribuição das variáveis numéricas
    - Countplot para verificar a distribuição das categorias
    - Heatmap de correlação para identificar relações entre variáveis
- Modelagem Inicial (Sem Tratamento)
    - Separação de variáveis preditoras e alvo
    - Split em treino e teste (train_test_split)
    - Implementação de modelos de regressão:
        - Linear Regression
        - Ridge Regression
        - Lasso Regression
        - Decision Tree Regressor
        - Random Forest Regressor
    - Avaliação dos modelos com métricas: MAE, RMSE, R²
- Pré-processamento
    - Identificação e remoção de outliers usando IQR
    - Tratamento da variável categórica com get_dummies (one-hot encoding)
- Modelagem Pós-Tratamento
    - Repetição do split em treino e teste após tratamento
    - Treinamento dos mesmos modelos de regressão
    - Avaliação dos modelos com as mesmas métricas
- Comparação de Resultados
    - Tabela comparativa das métricas antes e depois do tratamento dos dados


## Próximos Passos
 - ⏳ Implementação de Valização Cruzada
 - ⏳ Análise de Feature Importance
 - ⏳ Análise de Resíduos

## Conclusão:
⏳ Em andamento

## Referências

- [Kaggle Dataset - Electricity cost prediction dataset](https://www.kaggle.com/datasets/shalmamuji/electricity-cost-prediction-dataset/data?select=electricity_cost_dataset.csv)

---

*Projeto de estudos em Ciência de Dados - Lucas Lopes*