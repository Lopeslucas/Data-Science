# Car Insurance Data - Data Science Study

Este repositório contém um notebook de estudo e experimentação em Ciência de Dados, utilizando o conjunto de dados de seguro de automóveis disponível no Kaggle.

## Estrutura do Projeto

- **[Kaggle] - Car Insurance Data/**
  - `EDA_v1.ipynb`: Análise exploratória, tratamento de dados, engenharia de features e modelagem com RandomForest.
  - `Car_Insurance_Claim.csv`: Base de dados utilizada (não versionar se for grande/confidencial).

## Objetivos

- Explorar e entender o conjunto de dados.
- Realizar pré-processamento e análise exploratória.
- Construir e avaliar modelos de classificação para prever solicitações de seguro.
- Praticar técnicas de engenharia de features e otimização de modelos.

## Técnicas Utilizadas

- **Leitura e análise exploratória:**
    - Carregamento do dataset com pandas.
    - Análise de tipos, valores nulos e estatísticas descritivas.
    - Visualização de correlação (heatmap), boxplots e histogramas.
- **Tratamento de dados:**
    - Remoção de colunas irrelevantes (ID, POSTAL_CODE).
    - Conversão de tipos para variáveis binárias e inteiras.
    - Remoção de linhas com valores nulos.
- **Engenharia de features:**
    - Mapeamento manual de variáveis binárias e ordinais para valores numéricos.
- **Modelagem preditiva:**
    - Separação em variáveis preditoras (X) e alvo (y).
    - Divisão em treino e teste com estratificação.
    - Treinamento de RandomForestClassifier com ajuste de class_weight.
    - Avaliação com métricas: acurácia, precisão, recall, F1-score e classification_report.
    - Visualização da matriz de confusão.
    - Análise de importância das features.
    - Redução de features com base na importância e reavaliação do modelo.
- **Otimização de hiperparâmetros:**
    - GridSearchCV com validação cruzada para encontrar melhores parâmetros do RandomForest.

## Próximos Passos

- Testar outros modelos (Logistic Regression, XGBoost, SVM, etc).
- Implementar pipelines de pré-processamento e modelagem.
- Tratar desbalanceamento de classes com técnicas como SMOTE ou undersampling.
- Explorar técnicas de feature selection automática.
- Adicionar explicabilidade (SHAP/LIME) para interpretar as decisões do modelo.
- Realizar validação cruzada mais robusta e análise de overfitting.
- Documentar decisões e justificar escolhas de features/modelos/métricas.

## Como Executar

1. Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/seu-repo.git
    ```
2. Crie o ambiente conda:
    ```bash
    conda env create -f environment.yml
    ```
3. Ative o ambiente:
    ```bash
    conda activate nome-do-seu-ambiente
    ```
4. Abra o notebook no Jupyter ou VS Code.

## Referências

- [Kaggle Dataset](https://www.kaggle.com/datasets/sagnik1511/car-insurance-data)

---

*Projeto de estudos em Ciência de Dados