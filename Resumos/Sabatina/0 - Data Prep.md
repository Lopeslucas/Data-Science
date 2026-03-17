# DataPrep

## Por que não podemos fazer normalização antes do train_test_split?
Não devemos fazer normalização antes do train_test_split porque isso causa data leakage.

Métodos de normalização, como StandardScaler ou MinMaxScaler, calculam estatísticas dos dados, como média, desvio padrão, mínimo e máximo.

Se fizermos isso antes de separar treino e teste, essas estatísticas serão calculadas usando todo o dataset, incluindo o conjunto de teste, o que faz o modelo ter acesso indireto a informações que não deveria ter durante o treinamento.

O procedimento correto é fazer o split primeiro, depois ajustar o scaler apenas no conjunto de treino usando fit, e aplicar transform no treino e no teste.

## O que é pipeline e por que ele ajuda a evitar data leakage?
Pipeline é uma estrutura que permite encadear todas as etapas de pré-processamento e modelagem em um único fluxo, como encoding, normalização e treinamento do modelo.

Ele ajuda a evitar data leakage porque garante que todas as transformações sejam ajustadas apenas no conjunto de treino. Durante o fit, o pipeline aprende os parâmetros usando apenas os dados de treino, e depois aplica transform no conjunto de validação ou teste.

Isso é especialmente importante quando usamos validação cruzada ou grid search, porque o pipeline garante que cada fold faça o pré-processamento corretamente sem usar informação dos outros folds.

Sem pipeline, é fácil cometer data leakage ao aplicar normalização ou encoding antes do split.

## Por que one-hot encoding pode causar multicolinearidade?
One-hot encoding pode causar multicolinearidade porque ao transformar uma variável categórica com k categorias em k colunas binárias, criamos dependência linear entre elas.

Como a soma das colunas sempre será 1, uma coluna pode ser escrita como combinação das outras, o que gera multicolinearidade perfeita, conhecido como dummy variable trap.

Isso causa problemas principalmente em modelos lineares, pois torna os coeficientes instáveis.

Para evitar isso, removemos uma das colunas, usando k-1 dummies, por exemplo com drop_first no one-hot encoding.

## Qual a diferença entre padronização e normalização, e quando usar cada uma?
Normalização e padronização são técnicas de scaling usadas para colocar as variáveis na mesma escala.

Na padronização, transformamos os dados para terem média zero e desvio padrão igual a um, usando o Z-score.
Já na normalização, usamos min-max scaling para colocar os valores em um intervalo fixo, geralmente entre 0 e 1.

A normalização é útil quando queremos manter os valores dentro de um intervalo específico ou quando usamos algoritmos baseados em distância, como KNN e redes neurais.

A padronização é mais usada quando os algoritmos assumem dados centrados, como regressão, SVM e PCA.

Ambas são sensíveis a outliers, mas o MinMax é mais afetado, e nesses casos podemos usar RobustScaler.


## Por que OneHotEncoding pode ser um problema em alguns casos?
OneHotEncoding pode ser um problema quando a variável categórica tem muitas categorias, porque isso aumenta muito a dimensionalidade do dataset, criando muitas colunas.

Isso pode gerar sparsity, aumentar o custo computacional e prejudicar modelos que dependem de distância ou de muitas features.

Além disso, em modelos lineares pode ocorrer multicolinearidade se todas as categorias forem mantidas, por isso normalmente removemos uma coluna ou usamos drop_first.

Em casos com muitas categorias, alternativas como target encoding ou agrupamento de categorias raras podem ser mais adequadas.

Sparsity: 
- Sparsity (esparsidade) no contexto de One-Hot Encoding significa que a maior parte dos valores da matriz gerada são zeros.
- Uma matriz esparsa é uma matriz onde a grande maioria dos elementos são zero.

## Quando você NÃO deve usar OneHotEncoding?
Não devemos usar OneHotEncoding em alguns cenários específicos.

- Primeiro, quando a variável é ordinal, porque o OneHot perde a informação de ordem, sendo melhor usar encoding ordinal.
- Segundo, quando a variável tem muitas categorias, pois o OneHot aumenta muito a dimensionalidade do dataset, gerando sparsity e maior custo computacional.
- Também pode ser um problema em modelos lineares por causa da multicolinearidade, e em modelos baseados em distância, onde muitas dimensões podem prejudicar o cálculo da distância.

Nesses casos, alternativas como ordinal encoding, target encoding ou agrupamento de categorias podem ser mais adequadas.


## Como tratar dados desbalanceados?
Existem várias formas de tratar dados desbalanceados.

- Podemos usar oversampling, aumentando a quantidade da classe minoritária, ou undersampling, reduzindo a classe majoritária.
- Uma técnica comum é o SMOTE, que cria exemplos sintéticos da classe minoritária ao invés de apenas duplicar dados.
- Também podemos usar class_weight nos modelos, ajustar o threshold de decisão, ou escolher métricas adequadas como recall, F1 ou AUC ao invés de accuracy.

É importante aplicar essas técnicas apenas no conjunto de treino para evitar data leakage.


## Por que não devemos fazer scaling antes do train_test_split?
Não devemos fazer scaling antes do train_test_split porque o scaler calcula estatísticas como média e desvio padrão usando os dados.

Se fizermos isso antes da separação, essas estatísticas serão calculadas usando também os dados de teste, o que gera data leakage.

O correto é fazer o split primeiro, depois ajustar o scaler apenas no treino (fit) e aplicar a transformação no treino e no teste (transform).

Isso garante que o modelo não tenha acesso a informações do conjunto de teste durante o treinamento.

