# Outras Perguntas

## Overfitting x Underfitting
Overfitting ocorre quando o modelo se ajusta excessivamente aos dados de treino, aprendendo inclusive o ruído, o que resulta em baixa capacidade de generalização e pior desempenho no conjunto de teste.

Underfitting ocorre quando o modelo é simples demais para capturar o padrão dos dados, apresentando alto erro tanto no treino quanto no teste.

Em termos de bias e variance, underfitting está associado a alto bias e baixa variância, enquanto overfitting está associado a baixa bias e alta variância.

Para evitar overfitting podemos usar técnicas como regularização (L1/Lasso, L2/Ridge), validação cruzada, redução de features, aumento de dados e modelos menos complexos.

## Data leakage
Data leakage é quando o modelo tem acesso a informações que não estariam disponíveis no momento da predição, o que faz com que ele aprenda padrões irreais e tenha uma performance inflada.

Isso pode acontecer, por exemplo, quando eu uso dados do teste no treinamento, ou quando faço algum pré-processamento usando a base inteira antes de separar treino e teste. Pra evitar isso, eu sempre faço a divisão de treino e teste antes de qualquer transformação.

Além disso, uso pipeline pra garantir que os dados de treino são usados pra aprender as transformações. Ou seja, eu faço o fit no treino, e depois aplico o transform no teste. Assim eu evito que qualquer informação do teste influencie o modelo durante o treinamento.

Um exemplo clássico de leakage é calcular a média de uma variável usando toda a base antes de fazer imputação, porque isso já usa informação do teste.

- fit = aprende (ex: média, categorias, escala)
- transform = aplica o que foi aprendido

- scaler.fit(X_train)      # aprende média e desvio
- scaler.transform(X_test) # aplica no teste

## O que é overfitting por data leakage? Dê um exemplo real.
Overfitting por data leakage acontece quando o modelo aprende usando informações que não estariam disponíveis no momento da predição, fazendo com que ele tenha um desempenho muito bom no treino ou validação, mas pior em produção.

Um exemplo comum é fazer normalização antes de separar treino e teste, porque o scaler acaba usando informações de todo o dataset, inclusive do teste.

Outro exemplo é usar variáveis que contêm informação futura, como tentar prever inadimplência usando dados que só existem depois do cliente já ter atrasado.

Nesses casos o modelo parece muito bom, mas é porque ele aprendeu com dados que não deveria ter acesso.

## Qual a diferença entre overfitting por alta variância e overfitting por data leakage?
O overfitting por alta variância acontece quando o modelo é muito sensível aos dados de treino, ou seja, pequenas variações nos dados geram modelos muito diferentes. Isso geralmente ocorre quando o modelo é muito complexo, como árvores profundas, e ele acaba aprendendo ruído ao invés de padrão.

Já o overfitting por data leakage não está relacionado à complexidade do modelo, mas sim a um erro no processo. Ele acontece quando informações do teste ou do futuro vazam para o treino, fazendo o modelo aprender com dados que ele não deveria ter acesso.

Por exemplo, fazer imputação usando a base inteira antes da divisão já gera leakage. Então, enquanto a alta variância é um problema de generalização do modelo, o data leakage é um problema de preparação dos dados.


## Validação Holdout e Cruzada
Holdout é uma técnica simples onde dividimos os dados em treino, validação e teste. O conjunto de treino é usado para ajustar o modelo, o de validação para escolher hiperparâmetros, e o de teste para avaliação final.

Já a validação cruzada divide o conjunto de dados em K folds. O modelo é treinado em K-1 folds e testado no fold restante, repetindo o processo K vezes. A métrica final é a média dos resultados.

A validação cruzada é mais usada quando temos poucos dados, pois permite melhor estimativa de generalização e reduz variância da métrica.

O holdout é mais usado quando temos muitos dados ou quando o custo computacional é alto, pois é mais rápido.

## Por que não é uma boa ideia avaliar o modelo só com um único split de treino e teste?
Avaliar o modelo com um único split de treino e teste pode gerar uma avaliação pouco confiável, porque essa divisão pode não ser representativa dos dados. Isso significa que a performance pode variar bastante dependendo de como os dados foram divididos, gerando métricas instáveis ou até otimistas demais.

Como consequência, a gente pode escolher um modelo que não vai performar bem em produção. Por isso, utilizo validação cruzada, onde eu divido os dados em múltiplos folds e avalio o modelo em diferentes cenários. Isso reduz a variância da avaliação e me dá uma visão mais robusta da capacidade de generalização do modelo.

## Se mesmo com validação cruzada seu modelo ainda apresenta overfitting, o que você faria?
Se mesmo com validação cruzada o modelo continua com overfitting, eu entendo que o problema está na capacidade de generalização. Então eu começo atuando na complexidade do modelo, reduzindo profundidade, ajustando hiperparâmetros ou aplicando regularização.

Depois avalio as features. Posso fazer feature selection, remover variáveis redundantes ou altamente correlacionadas, e até aplicar redução de dimensionalidade, como PCA, se fizer sentido.

Também analiso a qualidade dos dados, como presença de ruído ou outliers que podem estar influenciando o modelo.

Além disso, avalio se faz sentido buscar mais dados, porque às vezes o problema é falta de representatividade. E por fim, posso testar outros modelos mais adequados ao problema, sempre validando se houve melhora na generalização e não só no treino.

## Como você detecta multicolinearidade e quais problemas ela causa no modelo?
Multicolinearidade ocorre quando duas ou mais variáveis explicativas estão fortemente correlacionadas, o que pode causar instabilidade nos coeficientes da regressão linear.

Podemos detectar multicolinearidade usando matriz de correlação, mas uma medida mais adequada é o VIF, Variance Inflation Factor, que indica quanto a variância de um coeficiente está inflada devido à correlação com outras variáveis.

Os principais problemas causados pela multicolinearidade são coeficientes instáveis, dificuldade de interpretação, aumento da variância dos estimadores e sinais inconsistentes.

Em alguns casos, a performance preditiva não é muito afetada, mas a interpretação do modelo fica comprometida.

Podemos tratar multicolinearidade removendo variáveis, combinando features ou usando regularização como Ridge ou Lasso.

## O que são hiperparâmetros e qual a diferença para parâmetros do modelo?
Os hiperparâmetros são valores definidos antes do treinamento e servem para controlar como o modelo vai aprender. Eles não são aprendidos diretamente dos dados, a gente precisa configurar eles manualmente ou usando técnicas como grid search, random search ou otimização bayesiana. A escolha dos hiperparâmetros pode impactar bastante no desempenho do modelo.

Já os parâmetros são valores internos do modelo que são aprendidos automaticamente durante o treinamento. Por exemplo, os coeficientes de uma regressão linear, os pesos de uma rede neural ou os centróides no K-means.

Esses parâmetros são ajustados pelo algoritmo para minimizar a função de erro ou maximizar a função objetivo.

## Por que não devemos usar o conjunto de teste para escolher hiperparâmetros?
O conjunto de teste deve ser usado apenas para a avaliação final do modelo, porque ele simula dados novos que o modelo nunca viu.

Se a gente usa o conjunto de teste para escolher hiperparâmetros, a gente acaba ajustando o modelo com base nesses dados, e isso gera data leakage.

Nesse caso, o modelo pode ficar com uma métrica muito boa no teste, mas quando for para produção ele perde desempenho, porque ele acabou se adaptando ao conjunto de teste.

O mais correto é separar em treino, validação e teste, onde o treino é usado para ajustar o modelo, a validação para escolher hiperparâmetros, e o teste apenas para a avaliação final.


## O que é cross-validation e por que usar?
Cross-validation é uma técnica de validação onde dividimos os dados em k folds e treinamos o modelo várias vezes, usando cada fold como validação e os outros como treino.

No final, calculamos a média das métricas para ter uma avaliação mais robusta do modelo.

Isso reduz a variância da avaliação e evita que o resultado dependa de uma única divisão de treino e validação.

É muito usada para ajuste de hiperparâmetros e seleção de modelos.


## O que acontece se o modelo tiver muito bias?
E se tiver muita variância?
- muito bias → underfitting: Modelo simples demais
- muita variância → overfitting: Modelo muito flexível
    
Bias alto acontece quando o modelo é muito simples e não consegue capturar o padrão dos dados, levando a underfitting.

Variância alta acontece quando o modelo é muito complexo e se ajusta demais aos dados de treino, capturando ruído e causando overfitting.

O objetivo é encontrar um equilíbrio entre bias e variância para que o modelo generalize bem.


## Qual a diferença entre overfitting causado por alta variância e overfitting causado por data leakage?
Overfitting por alta variância ocorre quando o modelo é muito complexo e acaba aprendendo não apenas os padrões reais, mas também o ruído dos dados, perdendo capacidade de generalização.

Já o overfitting causado por data leakage não está relacionado à complexidade do modelo, mas sim a um erro no processo de treinamento, onde informações que não deveriam estar disponíveis acabam vazando para o treino, fazendo com que o modelo tenha um desempenho artificialmente alto.