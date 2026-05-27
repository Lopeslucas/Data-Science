# EDA

---
# Pontos cobertos pelo questionario abaixo:
- Oque é um outlier ✅
- Tipos de gráficos x tipo de variavel ✅
- Tipos de ˜Missings˜ (MCAR, MAR e MNAR) ✅
- Medidas de Centralidade ✅
- Distribuição Assimetrica ✅
- Analise de variaveis Categoricas e Numericas ✅

---
## Explique como você começa um EDA do zero em uma base nova.
No início do EDA eu faço o entendimento do dataset, olhando colunas, tipos (numéricas, categóricas, datas), quantidade de linhas e identificando o target e a granularidade dos dados.

Também avalio valores ausentes, duplicidades e possíveis inconsistências.

Depois disso parto para análise univariada:
- Numéricas: uso histograma e boxplot para entender distribuição, simetria, presença de outliers e dispersão
- Categóricas: uso barplot e frequência para identificar distribuição e categorias raras

Em seguida faço análise bivariada:
- Numérica vs numérica: correlação e scatterplot
- Categórica vs numérica: boxplot por categoria

Por fim, aprofundo na análise de outliers usando técnicas como IQR, Z-score e análise por percentis, para entender quais valores estão fora do padrão e decidir o tratamento.

## O que é um outlier e como você decide se deve tratar ou não?
Outliers são valores atípicos que fogem do padrão da distribuição dos dados. Eles podem surgir por erro de coleta, erro de digitação ou podem ser eventos raros, mas válidos.

Pra identificar, eu posso usar visualizações como boxplot ou métodos estatísticos, como IQR ou desvio padrão.

A decisão de tratar ou não depende do contexto. Primeiro eu analiso se aquele valor faz sentido no problema. Por exemplo, uma idade negativa claramente é erro e deve ser tratada, mas uma idade de 90 anos pode ser um outlier válido.

Então, se for erro, eu trato ou removo. Se for um valor válido, eu avalio o impacto no modelo antes de decidir. E essa decisão é importante porque remover um outlier válido pode fazer o modelo perder informação relevante


## Qual tipo de gráfico usar para cada variável no EDA?
A escolha do gráfico depende tanto do tipo de variável quanto do objetivo da análise.

- Para variáveis numéricas, utilizo histograma para entender a distribuição, como simetria e caudas, e boxplot para identificar dispersão e outliers.
- Para variáveis categóricas, utilizo barplot para analisar frequência e identificar categorias dominantes ou raras.
- Em análise bivariada, utilizo scatterplot para visualizar a relação entre duas variáveis numéricas e matriz de correlação para medir a força dessa relação, sempre tomando cuidado com relações não lineares.
- Para categórica vs numérica, utilizo boxplot por categoria para comparar distribuições, e para categórica vs categórica, utilizo gráficos de frequência ou tabelas de contingência.

Então, a escolha do gráfico sempre depende do objetivo da análise, como entender distribuição, detectar outliers ou analisar relações entre variáveis.

## Como você trata missing values?
O tratamento de missing values começa entendendo o tipo de ausência
- se é completamente aleatória (MCAR)
- dependente de outras variáveis (MAR) 
- ou não aleatória (MNAR), que é o mais crítico. 

Em seguida, eu avalio o impacto da ausência e a quantidade de dados faltantes.

Para variáveis numéricas, geralmente utilizo mediana por ser mais robusta a outliers, e para categóricas utilizo moda ou crio uma categoria como “unknown”. Também posso remover linhas ou colunas dependendo do volume de missing.

Como diferencial, posso criar uma variável indicadora de missing, pois a ausência de informação pode ser relevante para o modelo. Sempre avaliando o impacto dessas decisões na performance do modelo.

## Como você analisa uma variável numérica em um dataset?
Quando eu vou analisar uma variável numérica, eu começo olhando estatísticas descritivas, como média, mediana, desvio padrão e quartis, pra entender a distribuição geral. Depois eu parto pra visualização. Eu uso histograma pra ver a distribuição dos dados, se ela é simétrica, se tem cauda longa, concentração de valores.

Também uso boxplot, que me ajuda a identificar outliers e entender a dispersão através dos quartis.

Além disso, posso fazer análise bivariada, por exemplo usando correlação ou scatter plot, pra ver como essa variável se relaciona com outras. Com isso eu consigo entender distribuição, dispersão, possíveis outliers e comportamento geral da variável.

E isso me ajuda a tomar decisões de pré-processamento, como tratar outliers ou aplicar transformação, se necessário.

## Qual a diferença entre média e mediana, e em que situação a média pode te enganar?
A média é a soma de todos os valores dividida pelo número de observações, enquanto a mediana é o valor central quando os dados estão ordenados.

A média pode enganar principalmente quando a distribuição tem outliers ou é assimétrica, porque esses valores extremos puxam a média pra cima ou pra baixo.

Já a mediana é mais robusta nesses casos, porque ela não é influenciada pelos extremos, então representa melhor o centro dos dados.

Por isso, em situações com outliers ou assimetria, a mediana costuma ser uma escolha melhor, inclusive pra imputação de valores faltantes.

## O que significa dizer que uma distribuição é assimétrica? E qual o impacto disso na análise?
Uma distribuição assimétrica é quando os dados não são equilibrados, ou seja, existe uma cauda mais longa pra um dos lados.

Se a cauda for pra direita, a gente chama de assimetria positiva, e se for pra esquerda, assimetria negativa. Isso normalmente indica que existe uma concentração maior de dados em uma região e alguns valores extremos puxando a distribuição.

O impacto disso é que métricas como a média podem ser distorcidas, porque esses valores extremos influenciam o resultado. Nesses casos, a mediana costuma representar melhor o centro dos dados.

Além disso, a assimetria pode indicar necessidade de transformação, dependendo do modelo que eu for usar.

## Como você analisa uma variável categórica? O que você procura entender nela?
Na análise de variável categórica, primeiro eu identifico se ela é nominal ou ordinal, porque isso influencia diretamente no tipo de tratamento que eu vou aplicar.

Depois eu analiso a distribuição das categorias, normalmente usando barplot ou frequência, pra entender quais categorias têm maior representatividade e quais são pouco frequentes.

Isso é importante porque categorias muito raras podem gerar ruído no modelo, então em alguns casos faz sentido agrupar em uma categoria ‘outros’.

Também posso fazer análise bivariada, por exemplo cruzando com a variável target, pra entender se alguma categoria tem mais impacto no problema.

Com isso eu consigo entender distribuição, relevância e possíveis ajustes que preciso fazer na variável.