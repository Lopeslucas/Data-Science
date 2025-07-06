# Breast Cancer Dataset - Data Science Study

Este repositório contém notebooks de estudo e experimentação em Ciência de Dados, utilizando o conjunto de dados de câncer de mama de Wisconsin (Kaggle).

## Estrutura do Projeto

**[Kaggle] - Breast Cancer Dataset/**
- `EDA-v1.ipynb`: Análise exploratória, tratamento de dados, seleção de features e modelagem com RandomForest.
- `breast-cancer.csv`: Base de dados utilizada (não versionar se for grande/confidencial).

## Objetivos

- ✅ Explorar e entender o conjunto de dados.
- ✅ Realizar pré-processamento e análise exploratória.
- ✅ Construir e avaliar modelos de classificação para prever câncer maligno/benigno.
- ✅ Comparar técnicas e métricas de avaliação.

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
    - Divisão em treino e teste (train_test_split, com stratify)
- Seleção de features:
    - RandomForestClassifier para calcular feature importance
    - Teste de performance variando o número de features (laço for + validação cruzada)
- Validação:
    - Validação cruzada (cross_val_score, 5-fold) para comparar diferentes conjuntos de features
- Avaliação de modelo:
    - Métricas: acurácia, classification_report, matriz de confusão
    - Curva ROC e cálculo da AUC
    - Análise de falsos positivos e falsos negativos
    - Foco em recall da classe 1 (maligno)
- Técnicas de balanceamento:
    - Aplicação do SMOTE para oversampling
    - Uso de class_weight='balanced' no RandomForest
    - Ajuste de threshold para maximizar recall
    - Comparação quantitativa entre as abordagens
- Visualização:
    - Gráficos de barras para feature importance
    - Gráfico de linha para desempenho vs. número de features
    - Curva ROC plotada
- Documentação e justificativa:
    - Discussão sobre trade-off entre sensibilidade e especificidade
    - Interpretação dos resultados

## Próximos Passos
- ✅ Trate desbalanceamento: Use técnicas como SMOTE, class_weight, ou ajuste de threshold.
- Use pipelines: Organize pré-processamento e modelagem em pipelines do sklearn.
- ✅ Explique resultados: Interprete as métricas, principalmente recall/precision para problemas críticos.
- ✅ Documente decisões: Justifique escolhas de features, modelos e métricas.
- Automatize: Pense em modularizar e automatizar etapas para reuso e escalabilidade.

## Conclusão:
Esse notebook foi um caso de estudo, onde eu consegui exercitar a parte da análise exploratória, uso do modelo de Classifição RandomForrest. Na reta final, eu acabei fazendo mais uso da IA para montar o fluxo de comparação dos modelos após aplicar técnicas de balanceamento dos dados e na parte de "conclusão" da Análise. Mesmo tendo feito uso da IA (Copilot), eu mapei alguns itens que eu devo procurar artigos para leitura complementar e entendimento:
- Conceito de Trade-off
- Uso do "predict_proba"
- Técnicas de balanceamento (smote, class_weight e Threshold)
- Pipeline de pré-processamento
- Modularização e automação das etapas

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

- [Kaggle Dataset - Breast Cancer Data](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)

---

*Projeto de estudos em Ciência de Dados - Lucas Lopes*