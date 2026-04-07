# EDA

## Qual tipo de gráfico usar para cada variável no EDA?
A escolha do gráfico depende tanto do tipo de variável quanto do objetivo da análise.

- Para variáveis numéricas, utilizo histograma para entender a distribuição, como simetria e caudas, e boxplot para identificar dispersão e outliers.
- Para variáveis categóricas, utilizo barplot para analisar frequência e identificar categorias dominantes ou raras.
- Em análise bivariada, utilizo scatterplot para visualizar a relação entre duas variáveis numéricas e matriz de correlação para medir a força dessa relação, sempre tomando cuidado com relações não lineares.
- Para categórica vs numérica, utilizo boxplot por categoria para comparar distribuições, e para categórica vs categórica, utilizo gráficos de frequência ou tabelas de contingência.

Então, a escolha do gráfico sempre depende do objetivo da análise, como entender distribuição, detectar outliers ou analisar relações entre variáveis.

## Como você trata missing values?

O tratamento de missing values começa entendendo o tipo de ausência, se é completamente aleatória (MCAR), dependente de outras variáveis (MAR) ou não aleatória (MNAR), que é o mais crítico. Em seguida, eu avalio o impacto da ausência e a quantidade de dados faltantes.

Para variáveis numéricas, geralmente utilizo mediana por ser mais robusta a outliers, e para categóricas utilizo moda ou crio uma categoria como “unknown”. Também posso remover linhas ou colunas dependendo do volume de missing.

Como diferencial, posso criar uma variável indicadora de missing, pois a ausência de informação pode ser relevante para o modelo. Sempre avaliando o impacto dessas decisões na performance do modelo.



