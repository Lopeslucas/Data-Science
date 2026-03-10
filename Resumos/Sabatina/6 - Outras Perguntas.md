# Outras Perguntas

## Overfitting x Underfitting
Overfitting ocorre quando o modelo se ajusta excessivamente aos dados de treino, aprendendo inclusive o ruído, o que resulta em baixa capacidade de generalização e pior desempenho no conjunto de teste.

Underfitting ocorre quando o modelo é simples demais para capturar o padrão dos dados, apresentando alto erro tanto no treino quanto no teste.

Em termos de bias e variance, underfitting está associado a alto bias e baixa variância, enquanto overfitting está associado a baixa bias e alta variância.

Para evitar overfitting podemos usar técnicas como regularização (L1/Lasso, L2/Ridge), validação cruzada, redução de features, aumento de dados e modelos menos complexos.

## Data leakage
Data leakage ocorre quando informações que não estariam disponíveis no momento da predição vazam para o conjunto de treino, fazendo com que o modelo aprenda padrões irreais e tenha desempenho superestimado durante a validação, mas falhe em produção.

Isso pode acontecer, por exemplo, quando fazemos normalização antes de separar treino e teste, quando usamos variáveis que contêm informação do futuro, quando fazemos target encoding sem validação cruzada, ou quando criamos features usando o target.

Em problemas de séries temporais, também ocorre leakage quando usamos dados futuros para prever o passado.

O resultado é um modelo com baixa capacidade de generalização e métricas irreais.

## O que é overfitting por data leakage? Dê um exemplo real.
Overfitting por data leakage acontece quando o modelo aprende usando informações que não estariam disponíveis no momento da predição, fazendo com que ele tenha um desempenho muito bom no treino ou validação, mas pior em produção.

Um exemplo comum é fazer normalização antes de separar treino e teste, porque o scaler acaba usando informações de todo o dataset, inclusive do teste.

Outro exemplo é usar variáveis que contêm informação futura, como tentar prever inadimplência usando dados que só existem depois do cliente já ter atrasado.

Nesses casos o modelo parece muito bom, mas é porque ele aprendeu com dados que não deveria ter acesso.

## Validação Holdout e Cruzada
Holdout é uma técnica simples onde dividimos os dados em treino, validação e teste. O conjunto de treino é usado para ajustar o modelo, o de validação para escolher hiperparâmetros, e o de teste para avaliação final.

Já a validação cruzada divide o conjunto de dados em K folds. O modelo é treinado em K-1 folds e testado no fold restante, repetindo o processo K vezes. A métrica final é a média dos resultados.

A validação cruzada é mais usada quando temos poucos dados, pois permite melhor estimativa de generalização e reduz variância da métrica.

O holdout é mais usado quando temos muitos dados ou quando o custo computacional é alto, pois é mais rápido.

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

