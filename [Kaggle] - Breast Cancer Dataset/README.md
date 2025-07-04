# Breast Cancer Dataset - Data Science Study

Este repositório contém notebooks de estudo e experimentação em Ciência de Dados, utilizando o conjunto de dados de câncer de mama de Wisconsin (Kaggle).

## Estrutura do Projeto

- **[Kaggle] - Breast Cancer Dataset/**
- `EDA-v1.ipynb`: Análise exploratória, tratamento de dados, seleção de features e modelagem com RandomForest.
- `breast-cancer.csv`: Base de dados utilizada (não versionar se for grande/confidencial).

## Objetivos

- Explorar e entender o conjunto de dados.
- Realizar pré-processamento e análise exploratória.
- Construir e avaliar modelos de classificação para prever câncer maligno/benigno.
- Comparar técnicas e métricas de avaliação.

## Técnicas Utilizadas

- Leitura e limpeza de dados com pandas.
- Análise exploratória (EDA) e visualizações com seaborn/matplotlib.
- Seleção de features com RandomForest.
- Validação cruzada e análise de performance.
- Avaliação com métricas como acurácia, matriz de confusão, classification report e curva ROC.

## Como Executar

1. Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/seu-repo.git
    ```
2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
3. Abra os notebooks no Jupyter ou VS Code.

## Próximos Passos

- Testar outros modelos (SVM, Logistic Regression, XGBoost, etc).
- Implementar pipelines de pré-processamento.
- Tratar desbalanceamento de classes.
- Ajustar hiperparâmetros com GridSearchCV.
- Adicionar explicabilidade (SHAP/LIME).

## Referências

- [Kaggle Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)

---

*Projeto de estudos em Ciência de Dados - Lucas Lopes*