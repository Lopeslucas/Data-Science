# Electricity Cost Prediction — Regressão Linear (Notebook)

Este repositório contém um notebook focado na implementação e análise de uma Regressão Linear aplicada ao dataset "Electricity Cost Prediction". O objetivo principal foi exercitar as nuances da Regressão Linear (diagnósticos, pressupostos e interpretação) — não a busca do "melhor modelo".

## Estrutura do projeto
- EDA-LinearRegression.ipynb — notebook com todo o fluxo: EDA, pré-processamento, treino do modelo linear, métricas e diagnóstico.
- electricity_cost_dataset.csv — base de dados utilizada.

## Objetivos
- Explorar e entender o dataset.
- Implementar e avaliar um modelo de Regressão Linear como baseline.
- Validar pressupostos do modelo (normalidade, linearidade, homocedasticidade, multicolinearidade).
- Documentar métricas e conclusões.

## Técnicas e etapas implementadas (no notebook)
- Leitura e inspeção inicial: head, info, describe, verificação de nulos e duplicados.
- Separação de variáveis numéricas e categóricas.
- Visualizações:
  - Boxplot, Violinplot e Histplot para análise de distribuição e outliers.
  - Countplot para distribuição de categorias.
- Tratamento de variável categórica:
  - Codificação ordinal para `structure type` (com ordem definida explicitamente).
- Modelagem:
  - Separação treino/teste (train_test_split).
  - Treinamento com sklearn.linear_model.LinearRegression.
  - Predição em conjunto teste.
- Avaliação do modelo:
  - Métricas: MAE, MSE, RMSE, R².
- Scatter plot Real x Previsto e histograma de resíduos.
- Cálculo de VIF (multicolinearidade) com constante adicionada.
- Diagnósticos e validação de pressupostos (statsmodels):
  - Ajuste OLS (statsmodels) para obter resíduos e fitted values.
  - Gráfico Resíduos × Valores Ajustados (linearidade/heterocedasticidade).
  - Q–Q plot dos resíduos (normalidade).


## Principais achados (resumido)
- Regressão Linear apresentou desempenho consistente nas métricas (boa baseline).
- Resíduos aproximam-se razoavelmente da normalidade (Q–Q plot), porém há indícios de heterocedasticidade no gráfico resíduos × ajustados.

## Recomendações / próximos passos
- Validar estabilidade das métricas com validação cruzada.
- Se heterocedasticidade persistir: testar transformação da variável-alvo (log), WLS ou uso de erros-robustos para inferência.
- Manter a Regressão Linear como baseline; documentar próximos experimentos fora do escopo atual.

## Conclusão
A Regressão Linear é um bom exercício para entender pressupostos e diagnóstico de modelos de regressão neste dataset. Para produção/decisão, recomenda-se validação adicional e investigação sobre heterocedasticidade e encoding categórico.

## Referência
- Dataset: https://www.kaggle.com/datasets/shalmamuji/electricity-cost-prediction-dataset/data?select=electricity_cost_dataset.csv

*Projeto de estudos em Ciência de Dados — Lucas Lopes*