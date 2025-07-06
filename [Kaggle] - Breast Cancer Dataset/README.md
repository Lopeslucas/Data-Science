# Breast Cancer Dataset - Data Science Study

Este repositório contém notebooks de estudo e experimentação em Ciência de Dados, utilizando o conjunto de dados de câncer de mama de Wisconsin (Kaggle).

## Estrutura do Projeto

**[Kaggle] - Breast Cancer Dataset/**
- `EDA-v1.ipynb`: Análise exploratória, tratamento de dados, seleção de features e modelagem com RandomForest.
- `breast-cancer.csv`: Base de dados utilizada (não versionar se for grande/confidencial).

## Objetivos

- Explorar e entender o conjunto de dados.
- Realizar pré-processamento e análise exploratória.
- Construir e avaliar modelos de classificação para prever câncer maligno/benigno.
- Comparar técnicas e métricas de avaliação.

## Técnicas Utilizadas

- Leitura e exploração de dados:
    - Leitura do CSV com pandas
    - Análise de tipos de dados, valores nulos e duplicados
    - Estatísticas descritivas com describe()
    - Verificação de balanceamento da variável alvo
- Pré-processamento:
    - Remoção de coluna irrelevante (ID)
    - Conversão da variável alvo de categórica para numérica (map B/M para 0/1)
- Análise exploratória:
    - Heatmap de correlação entre variáveis numéricas (seaborn)
- Modelagem:
    - Separação em variáveis preditoras (X) e alvo (y)
    - Divisão em treino e teste (train_test_split)
- Seleção de features:
    - RandomForestClassifier para calcular feature importance
    - Teste de performance variando o número de features (laço for + validação cruzada)
- Validação:
    - Validação cruzada (cross_val_score, 5-fold) para comparar diferentes conjuntos de features
- Avaliação de modelo:
    - Métricas: acurácia, classification_report, matriz de confusão
    - Curva ROC e cálculo da AUC
- Visualização:
    - Gráficos de barras para feature importance
    - Gráfico de linha para desempenho vs. número de features
    - Curva ROC plotada


## Próximos Passos
- Trate desbalanceamento: Use técnicas como SMOTE, class_weight, ou ajuste de threshold.
- Teste múltiplos modelos: Compare diferentes algoritmos e escolha o melhor via validação cruzada.
- Ajuste de hiperparâmetros: Use GridSearchCV ou RandomizedSearchCV para buscar os melhores parâmetros.
- Use pipelines: Organize pré-processamento e modelagem em pipelines do sklearn.
- Validação robusta: Sempre reporte resultados com cross-validation, não só holdout.
- Explique resultados: Interprete as métricas, principalmente recall/precision para problemas críticos.
- Documente decisões: Justifique escolhas de features, modelos e métricas.
- Automatize: Pense em modularizar e automatizar etapas para reuso e escalabilidade.
- Considere Explainability: Use SHAP, LIME ou feature importance para explicar decisões do modelo.
- Teste em dados externos: Se possível, valide o modelo em outro dataset para garantir generalização.

## Erros e Pontos de Atenção
- Desbalanceamento: Você notou o desbalanceamento, mas não aplicou técnicas para tratá-lo (ex: oversampling, undersampling, SMOTE, ou class_weight).
- Escalonamento: Não aplicou padronização/normalização nas features, o que pode ser importante para alguns modelos (menos crítico para RandomForest, mas essencial para SVM, KNN, etc.).
- Validação cruzada só na seleção de features: O modelo final foi avaliado só no holdout (train_test_split). O ideal é usar cross-validation também na etapa final para evitar overfitting.
- Threshold fixo: Usou o threshold padrão (0.5) para classificação. Poderia ajustar para reduzir falsos negativos, especialmente em problemas de saúde.
- Pouca experimentação de modelos: Só testou RandomForest. Poderia comparar com outros modelos (SVM, Logistic Regression, XGBoost, etc.).
- Sem análise de variáveis categóricas: Aqui não impactou, mas em outros projetos é fundamental.
- Sem pipeline: Não usou sklearn.pipeline para organizar pré-processamento + modelo, o que é boa prática para reprodutibilidade.

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

## Referências

- [Kaggle Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)

---

*Projeto de estudos em Ciência de Dados - Lucas Lopes*