## Por que árvore de decisão não precisa de normalização, mas regressão precisa?
Modelos de árvore de decisão não precisam de normalização porque eles fazem divisões baseadas em regras do tipo feature menor que um determinado valor. Essas divisões dependem apenas da ordenação dos valores e não da escala, então não importa se uma variável está entre 0 e 100 ou entre 0 e 10000.

Já modelos lineares e modelos baseados em otimização, como regressão, SVM e KNN, são sensíveis à escala porque os coeficientes e o processo de treinamento dependem da magnitude das variáveis. Se as variáveis estiverem em escalas muito diferentes, o modelo pode dar mais peso para variáveis com valores maiores, além de dificultar a convergência do algoritmo.

Por isso, nesses modelos é comum usar normalização ou padronização, enquanto em árvores isso não é necessário.


## Qual a diferença entre Random Forest e Gradient Boosting?
Random Forest é um método de ensemble baseado em Bagging, onde várias árvores são treinadas de forma independente usando bootstrap sampling dos dados.

Além disso, em cada split é usado um subconjunto aleatório de features, o que reduz a correlação entre as árvores e diminui a variância do modelo. A previsão final é feita pela média ou voto das árvores.

Já o Gradient Boosting é um método sequencial, onde cada nova árvore é treinada para corrigir os erros da anterior, otimizando gradualmente a função de perda através do gradiente.
Isso reduz o viés e permite aprender padrões complexos, mas torna o modelo mais sensível a overfitting e aos hiperparâmetros.

Em geral, Random Forest é mais robusto e fácil de usar, enquanto Gradient Boosting costuma ter melhor performance, mas exige mais tuning.


## Em quais situações um modelo linear pode performar melhor que uma árvore de decisão?
Modelos lineares podem performar melhor que árvores quando a relação entre as variáveis é aproximadamente linear, porque nesse caso o modelo consegue capturar o padrão sem precisar de muita complexidade.

Eles também costumam funcionar melhor em datasets menores, porque têm menor variância e menos risco de overfitting, enquanto árvores de decisão tendem a crescer muito e se ajustar demais aos dados de treino.

Outra vantagem dos modelos lineares é a interpretabilidade, já que os coeficientes mostram diretamente a influência de cada variável.

Além disso, modelos lineares costumam ter menor custo computacional e funcionam bem quando os dados não têm muita não-linearidade ou ruído.

Já árvores são mais indicadas quando existem interações complexas entre variáveis ou relações não lineares.


## Qual a diferença de viés e variância entre modelos lineares, árvores de decisão e Random Forest?
Modelos lineares costumam ter alto viés e baixa variância, porque eles assumem uma relação linear entre as variáveis, então são mais simples e menos sensíveis a variações nos dados, o que reduz overfitting, mas pode causar underfitting quando os dados são mais complexos.

Árvores de decisão têm baixo viés e alta variância, porque conseguem se ajustar muito bem aos dados de treino, capturando padrões complexos, mas pequenas mudanças nos dados podem gerar árvores muito diferentes, o que aumenta o risco de overfitting.

Já o Random Forest reduz a variância da árvore de decisão usando bagging, com bootstrap e várias árvores independentes, e depois combinando as previsões pela média ou voto.

Com isso, o Random Forest fica com variância menor que a árvore simples e um viés um pouco maior, mas com melhor capacidade de generalização.


## Por que aumentar a profundidade da árvore aumenta variância, mas aumentar o número de variáveis em um modelo linear aumenta risco de overfitting?
Em modelos de árvore, quando aumentamos muito a profundidade, a árvore começa a fazer divisões com poucas amostras em cada nó, às vezes até com uma única observação.

Isso faz com que o modelo fique muito sensível aos dados de treino, aumentando a variância e causando overfitting, porque ele passa a aprender o ruído em vez do padrão real. Já em modelos lineares, o risco de overfitting aumenta quando adicionamos muitas variáveis, porque o modelo fica mais complexo e passa a ter mais parâmetros para ajustar.

Isso pode causar multicolinearidade e deixar os coeficientes instáveis, além de aumentar a variância do modelo, fazendo com que ele se ajuste demais aos dados de treino. Por isso, em modelos lineares é comum usar regularização ou seleção de variáveis para controlar a complexidade e evitar overfitting.


## Por que Random Forest tem custo computacional maior que árvore de decisão, e como podemos reduzir esse custo usando hiperparâmetros?
O Random Forest tem custo computacional maior que uma árvore de decisão porque ele constrói várias árvores usando bagging, com bootstrap e seleção aleatória de features, enquanto a árvore de decisão constrói apenas uma árvore.

Como o Random Forest pode ter dezenas ou centenas de árvores, o tempo de treino e o uso de memória aumentam bastante, principalmente quando as árvores são profundas. A gente consegue reduzir esse custo usando hiperparâmetros como n_estimators, que controla o número de árvores, max_depth, que limita a profundidade, e max_features, que define quantas variáveis são usadas em cada split.

Reduzindo esses valores, o modelo fica mais leve, mas também pode perder performance, então existe um trade-off entre custo computacional e capacidade de generalização. Uma vantagem do Random Forest é que as árvores são independentes, então o treino pode ser paralelizado, o que ajuda a reduzir o tempo de execução.


## Por que a padronização ajuda o Gradient Descent a convergir mais rápido em modelos lineares?
A padronização ajuda o Gradient Descent a convergir mais rápido porque em modelos lineares nós temos variáveis que podem estar em escalas muito diferentes.

Quando uma variável tem valores muito maiores que outra, ela acaba influenciando mais o cálculo do gradiente, fazendo com que o algoritmo demore mais para encontrar o mínimo da função de erro.

Isso faz com que o Gradient Descent siga um caminho irregular, em forma de zig-zag, e a convergência fique mais lenta.

Quando fazemos a padronização, os dados passam a ter média zero e desvio padrão igual a um, deixando todas as variáveis na mesma escala, o que facilita o cálculo dos coeficientes e faz com que o Gradient Descent converja mais rápido e de forma mais estável.


## Por que modelos de árvore tendem a ter alta variância e modelos lineares tendem a ter alto viés?
- árvore → muito flexível → baixa bias / alta variância
- linear → pouco flexível → alto bias / baixa variância

Modelos baseados em árvore tendem a ter alta variância porque eles são muito flexíveis.

Como a árvore faz divisões usando thresholds e escolhe o melhor split em cada passo, pequenas mudanças nos dados de treino podem gerar árvores completamente diferentes. Isso faz com que o modelo se ajuste muito aos dados de treino, aumentando o risco de overfitting, por isso dizemos que árvores têm alta variância e baixo viés.

Já modelos lineares tendem a ter alto viés porque eles assumem uma relação linear entre as variáveis, então o modelo tem menos flexibilidade para se adaptar a padrões mais complexos. Isso faz com que ele generalize melhor e tenha menor variância, mas pode cometer mais erro se a relação real dos dados não for linear.

Então existe um trade-off entre viés e variância, onde modelos mais simples têm mais viés e menos variância, e modelos mais complexos têm menos viés e mais variância.
