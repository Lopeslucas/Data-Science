# Modelos Baseados em Árvores + Ensembles

Os métodos baseados em árvores envolvem a estratificação ou segmentação do espaço de preditores em uma série de regiões simples, cujas regras de divisão podem ser resumidas em um diagrama de árvore. Já os métodos ensemble são abordagens que combinam vários desses modelos simples (conhecidos como "aprendizes fracos") para obter um único modelo preditivo muito mais potente e robusto.

1. Conceito e Funcionamento
- Árvores de Decisão: O processo começa com a divisão binária recursiva, um algoritmo top-down e "ganancioso" (greedy) que divide o espaço de dados em regiões retangulares buscando minimizar o erro (RSS para regressão e o Índice Gini ou Entropia para classificação). Para uma nova observação, o modelo prevê a média (regressão) ou a moda (classificação) das observações de treinamento na região onde ela se encontra.
- Modelos Ensemble: Como árvores individuais podem ser instáveis, utilizam-se técnicas para agregá-las:
- Bagging: Constrói múltiplas árvores de forma independente em amostras repetidas (bootstrap) e tira a média das previsões para reduzir a variância.
    - Random Forests: Melhora o bagging ao considerar apenas um subconjunto aleatório de preditores em cada divisão, o que descorrelaciona as árvores e torna o modelo mais confiável.
    - Boosting: As árvores são construídas sequencialmente, onde cada nova árvore aprende com os erros (resíduos) das anteriores, resultando em um aprendizado lento e preciso.
    - BART: Uma técnica bayesiana que gera novas árvores através de pequenas perturbações aleatórias nas existentes, explorando o espaço do modelo de forma eficaz.

2. Premissas Principais
- Particionamento Retangular: O método assume que o relacionamento entre as variáveis pode ser adequadamente capturado ao dividir o espaço de preditores em caixas ou retângulos de alta dimensão.
- Modelo Constante por Partes: Assume-se que, dentro de cada região final (folha), a resposta pode ser representada de forma satisfatória por um único valor constante.
- Independência no Bagging/Random Forests: Pressupõe-se que a média de muitos modelos independentes ou descorrelacionados terá uma variância menor do que um único modelo.

3. Vantagens
- Interpretabilidade: Árvores simples são extremamente fáceis de explicar e podem ser visualizadas graficamente, sendo às vezes mais intuitivas que a regressão linear.
- Espelhamento Humano: Acredita-se que as árvores mimetizam o processo de tomada de decisão humana de forma mais próxima que outros métodos.
- Tratamento de Dados: Elas lidam facilmente com preditores qualitativos sem a necessidade de criar variáveis dummy (embora algumas implementações de software ainda as exijam).
- Alta Precisão (Ensembles): Quando combinadas em florestas ou via boosting, alcançam níveis de precisão competitivos com os melhores métodos de aprendizado supervisionado.

4. Desvantagens
- Instabilidade (Single Trees): Árvores individuais sofrem de alta variância, o que significa que uma pequena mudança nos dados de treinamento pode gerar uma árvore completamente diferente.
- Menor Precisão Base: Uma árvore de decisão isolada geralmente não possui o mesmo poder preditivo que modelos lineares ou outras abordagens clássicas.
- Perda de Interpretabilidade (Ensembles): Embora ensembles melhorem a precisão, eles transformam o modelo em uma "caixa preta", tornando impossível representar o processo de decisão em um único diagrama simples.
- Risco de Overfitting (Boosting): Diferente do Bagging e Random Forests, o Boosting pode sofrer sobreajuste se o número de árvores for excessivamente grande, exigindo ajuste cuidadoso.

---
# Pontos cobertos pelo questionario abaixo:
Árvores
- Como funciona o Threshold de escolha do melhor corte + no raiz, da arvore de decisão ✅
- Como funciona o corte de Threhold em variaveis categoricas e numericas ✅
- Sensibilidade das arvores a sofrerem com overfiting e como Random Forrest lida ✅
- Como é medido o corte das arvores, com metricas de Gini e Entropia ✅
- Tecnicas de Pre e Pos Poda ✅
    - Ajuste de Hiperparametros ✅

Ensemble (Random Forrest + Tecnica de Bagging x Boosting)
- Como funciona a contrução do Random Forrest com tecnica de Bagging ✅
- Como Random Forrest lida com o Overfiting ✅
- O que é o OOB (Out Of Bag) ✅
- O que é o Learning Rate no Boosting ✅
- Quais as principais diferenças entre Bagging x Boosting ✅
- Por que o Boosting é sensivel ao Overfiting ✅
- Relação Vies Variancia nos modelos Ensemble ✅

---
# Árvores 

## Como a árvore de decisão escolhe o melhor split? (critério de divisão)
A árvore de decisão escolhe o melhor split avaliando todas as possíveis divisões das variáveis e selecionando aquela que maximiza a separação dos dados naquele nó.

- Para problemas de classificação, os critérios mais usados são Gini impurity e Entropy, que medem o grau de impureza das classes. O split escolhido é aquele que reduz mais a impureza, ou seja, que gera maior information gain.
- Para regressão, normalmente usamos redução de variância ou MSE para escolher o split.

A árvore é considerada um algoritmo guloso porque ela escolhe sempre a melhor divisão local em cada nó, sem garantir que isso leve à melhor solução global.


## Como uma árvore de decisão começa a ser construída e como o algoritmo escolhe o nó raiz?
A árvore de decisão começa a ser construída a partir do nó raiz, onde o algoritmo testa todas as possíveis divisões das variáveis para encontrar o melhor split.

Esse melhor split é escolhido com base na minimização de uma função de custo. No caso de classificação, normalmente são usadas métricas como Gini ou entropia, que medem o nível de impureza dos dados. O algoritmo escolhe a divisão que deixa os grupos mais puros, ou seja, com mais elementos da mesma classe.

Para regressão, o critério costuma ser erro quadrático médio ou variância, buscando a divisão que reduz mais o erro. A partir do nó raiz, o processo continua de forma recursiva, criando novos nós de decisão até chegar nos nós folha.

Esse processo é guloso, porque em cada passo ele escolhe a melhor divisão naquele momento, sem olhar para o futuro.


## Como o algoritmo testa os possíveis splits em variáveis numéricas e categóricas dentro de uma árvore de decisão?
Dentro da árvore de decisão, o algoritmo testa vários possíveis splits para cada variável para encontrar a melhor divisão.

- Para variáveis numéricas, ele testa vários pontos de corte, chamados de threshold, por exemplo idade menor que 30, menor que 40, menor que 50, e assim por diante, calculando a função de custo para cada divisão.
- Para variáveis categóricas, ele testa diferentes agrupamentos das categorias, tentando separar os dados da forma que reduza mais a impureza no caso de classificação, ou o erro no caso de regressão.

Depois de testar todos os splits possíveis, o algoritmo escolhe aquele que minimiza a função de custo, e usa esse como a divisão do nó.


## Por que árvores de decisão tendem a overfitting? E como Random Forest reduz isso?
Árvores de decisão tendem a overfitting porque são modelos de baixa bias e alta variância. Elas podem crescer até separar completamente os dados de treino, aprendendo inclusive o ruído, o que reduz a capacidade de generalização.

O Random Forest reduz esse problema usando bagging, onde várias árvores são treinadas em amostras bootstrap diferentes do dataset, e a previsão final é feita pela média ou votação. Além disso, em cada split o Random Forest usa apenas um subconjunto aleatório de features, o que reduz a correlação entre as árvores.

Essa combinação reduz a variância do modelo e diminui o risco de overfitting, tornando o Random Forest mais robusto que uma única árvore.


## O que acontece se eu aumentar muito a profundidade de uma árvore? E como controlar isso?
Modelos baseados em árvore, quando têm profundidade muito alta, tendem a sofrer overfitting, porque a árvore continua fazendo divisões até separar quase perfeitamente os dados de treino, e com isso ela acaba capturando o ruído ao invés de aprender só o padrão.

Para controlar isso, a gente pode limitar a profundidade da árvore usando max_depth, definir um número mínimo de amostras para dividir com min_samples_split ou min_samples_leaf, e também limitar o número de folhas.

Essas técnicas ajudam a reduzir a variância do modelo e evitam que ele fique muito ajustado aos dados de treino.


## O que é Gini e Entropy e qual a diferença entre eles na árvore de decisão?
- Impureza = mistura de classes
- Pureza = só uma classe

Gini e Entropy são métricas usadas na árvore de decisão para medir a impureza dos nós, ou seja, o quanto os dados estão misturados entre as classes. O algoritmo testa vários splits e escolhe aquele que reduz mais a impureza.

O Gini mede a probabilidade de classificar um elemento errado dentro do nó, então quanto menor o Gini, mais puro é o nó.

Já a Entropy mede o nível de desordem dos dados usando conceito de informação, sendo mais sensível, mas também mais custosa computacionalmente porque usa log.

Na prática, os dois costumam gerar resultados parecidos, mas o Gini é mais usado porque é mais rápido.


## Em árvore de decisão para regressão, qual função de custo é usada para escolher o split?
Em árvore de decisão para regressão, o critério mais usado para escolher o split é o erro quadrático médio, o MSE.

O algoritmo testa várias divisões e escolhe aquela que reduz mais o erro dentro dos nós, ou seja, que deixa os valores mais próximos entre si.

Na prática, isso significa reduzir a variância dentro de cada nó, fazendo com que os dados fiquem mais homogêneos.

Além do MSE, também pode ser usado MAE, mas o MSE é mais comum porque penaliza mais erros grandes.


## Por que árvores de decisão tendem a overfitting e como funciona a pré-poda e a pós-poda para evitar isso?
Árvores de decisão tendem a sofrer overfitting porque elas têm alta variância e vão crescendo até se ajustar muito aos dados de treino.

Conforme a árvore aumenta a profundidade, ela pode chegar em nós com poucas amostras, às vezes até uma só, e nesse caso ela começa a aprender o ruído dos dados em vez do padrão real, perdendo capacidade de generalização.

A pré-poda é quando a gente limita o crescimento da árvore antes dela ficar muito complexa, usando hiperparâmetros que controlam o tamanho da árvore, como 
- max_depth: Define a profundidade máxima da árvore.
- min_samples_split: Número mínimo de amostras necessárias para dividir um nó.
- min_samples_leaf: Número mínimo de amostras que cada folha precisa ter.

Já a pós-poda acontece depois que a árvore já foi construída, removendo alguns ramos que não trazem ganho real na divisão, deixando o modelo mais simples e reduzindo o overfitting.

Essas técnicas ajudam a controlar a complexidade da árvore e melhorar a generalização.


## Como árvores de decisão fazem split em variáveis categóricas e variáveis numéricas?
A árvore de decisão faz os splits avaliando todas as possíveis divisões das variáveis e escolhendo aquela que minimiza a função de custo naquele momento.

Para variáveis numéricas, o split é feito usando um threshold, ou seja, a árvore testa valores do tipo menor ou igual e maior que um determinado valor, e escolhe o corte que gera a melhor separação dos dados, usando métricas como Gini ou Entropia para classificação, ou MSE para regressão.

Já para variáveis categóricas, o split pode ser feito separando grupos de categorias, por exemplo, pertence a um conjunto de categorias ou não pertence. A árvore testa diferentes combinações possíveis e escolhe a divisão que gera maior pureza nos nós.

Na maioria das implementações, como no sklearn, a árvore faz splits binários, então mesmo para variável categórica ela divide em dois grupos. Dependendo do algoritmo, pode ser necessário fazer encoding antes, mas modelos de árvore costumam ser mais flexíveis com variáveis categóricas do que modelos lineares.


## Por que modelos de árvore não precisam de scaling?
Modelos baseados em árvore não precisam de scaling porque eles não utilizam métricas de distância ou gradientes diretamente sobre as features.

O algoritmo constrói a árvore realizando divisões baseadas em thresholds em cada variável, buscando a divisão que maximiza o ganho de informação ou reduz a impureza (como Gini ou MSE).

Como cada feature é avaliada individualmente, a escala absoluta da variável não influencia a escolha da divisão.

Diferente de modelos baseados em distância, como KNN ou SVM, onde features com escalas maiores podem dominar o cálculo da distância.


## Por que árvores tendem a overfitar facilmente?
Árvores de decisão tendem a overfitar porque o algoritmo constrói a árvore de forma greedy, escolhendo a melhor divisão local em cada passo para reduzir a impureza.

Se não houver restrições, a árvore continua dividindo até que os nós fiquem com poucas amostras, ou até mesmo uma única observação, o que faz com que o modelo capture ruído dos dados de treino.

Isso torna a árvore um modelo de alta variância, pois pequenas mudanças nos dados podem gerar árvores completamente diferentes.

Por isso é necessário controlar a complexidade usando parâmetros como max_depth, min_samples_leaf e min_samples_split.



---
# Ensemble 
## Por que Random Forest costuma overfitar menos que uma árvore de decisão?
O Random Forest costuma sofrer menos overfitting porque ele treina várias árvores independentes usando amostras diferentes dos dados, através de bootstrap, e depois faz a predição pela média ou votação.

Além disso, em cada divisão ele usa apenas um subconjunto aleatório de features, o que reduz a correlação entre as árvores. Isso faz com que o modelo tenha menor variância e fique mais robusto, reduzindo o risco de overfitting quando comparado com uma única árvore de decisão, que tende a se ajustar muito aos dados de treino.

Em contrapartida, o Random Forest tem um custo computacional maior e é menos interpretável do que uma árvore simples.


## Se árvore de decisão tem alta variância e sofre overfitting, como o Random Forest resolve esse problema?
O Random Forest reduz o overfitting porque ele usa a técnica de bagging, que é o bootstrap com aggregation.

Ele constrói várias árvores independentes usando amostras diferentes dos dados, geradas por bootstrap, e depois junta as previsões, usando voto majoritário na classificação ou média na regressão.

Além disso, em cada split ele usa apenas um subconjunto aleatório de features, o que reduz a correlação entre as árvores.

Como as árvores ficam menos correlacionadas, a média das previsões reduz a variância do modelo, deixando ele mais estável e com menor risco de overfitting.

Alguns hiperparâmetros importantes no Random Forest são:
- n_estimators: Define quantas árvores o modelo vai criar.
- max_depth: Define a profundidade máxima de cada árvore.
- max_features: Define quantas variáveis (features) podem ser consideradas em cada divisão da árvore.


## O que é Out Of Bag (OOB) no Random Forest e para que ele serve?
O Random Forest, como usamos bootstrap, cada árvore é treinada com uma amostra aleatória dos dados com reposição. Isso faz com que algumas amostras não sejam usadas no treinamento de uma determinada árvore. Essas amostras são chamadas de Out Of Bag, ou OOB.

A gente pode usar esses dados que ficaram de fora para avaliar o desempenho da árvore, funcionando como uma validação interna, sem precisar separar um conjunto de teste. O OOB é útil para estimar o erro do modelo e verificar a capacidade de generalização, sendo muito usado em Random Forest.
- Bootstrap → cada árvore vê dados diferentes
- Dados que ficaram fora → OOB
- Usa OOB → como se fosse teste


## O que é learning rate no Gradient Boosting e o que acontece se ele for muito alto ou muito baixo?
O learning rate no Gradient Boosting controla o quanto cada nova árvore influencia no modelo final. Como o boosting cria árvores sequenciais, cada árvore corrige o erro da anterior, e o learning rate define o tamanho desse ajuste. Se o learning rate for muito alto, o modelo aprende rápido demais e pode sofrer overfitting.

Se for muito baixo, o modelo aprende mais devagar e precisa de mais árvores, mas geralmente generaliza melhor. Por isso, o learning rate é um hiperparâmetro importante para controlar o equilíbrio entre bias e variância.


## Qual a diferença entre Bagging e Boosting?
O bagging é um método onde a gente treina várias árvores independentes usando amostras diferentes dos dados, geralmente com bootstrap, e depois faz a predição pela média ou votação. Isso ajuda a reduzir a variância e deixa o modelo mais estável.

Já o boosting funciona de forma sequencial, onde cada nova árvore é treinada para corrigir o erro da anterior. Com isso o modelo vai melhorando aos poucos, reduzindo o viés e conseguindo aprender padrões mais complexos.

Um exemplo de bagging é o Random Forest, e de boosting são modelos como Gradient Boosting, XGBoost e LightGBM.


## Qual a diferença principal entre Bagging e Boosting na forma como as árvores são treinadas?
A técnica de bagging funciona usando bootstrap, onde várias árvores são treinadas de forma independente com amostras diferentes dos dados. 
- Como as árvores são independentes, o modelo final faz a predição pela média na regressão ou voto majoritário na classificação, o que reduz a variância e diminui o risco de overfitting.
- Já o boosting funciona de forma sequencial, onde cada nova árvore é treinada para corrigir os erros da anterior.

Isso faz com que o modelo reduza o viés e consiga aprender padrões mais complexos, mas também deixa o modelo mais sensível a overfitting se não controlar bem os hiperparâmetros.

Alguns hiperparâmetros importantes no boosting são:
- n_estimators: Define quantas árvores serão construídas no modelo.
- learning_rate: Controla o quanto cada árvore contribui para o modelo final.
- max_depth: Define a profundidade máxima de cada árvore usada no boosting.


## Por que Gradient Boosting costuma ter melhor performance que Random Forest, mas é mais sensível a overfitting?
O Gradient Boosting costuma ter melhor performance que o Random Forest porque ele constrói as árvores de forma sequencial, onde cada nova árvore é treinada para corrigir os erros da anterior. Com isso, o modelo vai minimizando o erro gradualmente, aprendendo padrões mais complexos e reduzindo o viés, enquanto o Random Forest trabalha com árvores independentes e paralelas. Porém, por ser sequencial e mais complexo, o Gradient Boosting fica mais sensível a overfitting se os hiperparâmetros não forem bem ajustados.

Alguns hiperparâmetros importantes são o learning_rate, que controla o quanto cada árvore influencia no modelo, o n_estimators, que define quantas árvores serão criadas, e o max_depth, que controla o tamanho das árvores.

Se esses valores forem muito altos, o modelo pode se ajustar demais aos dados de treino e perder capacidade de generalização, por isso o boosting costuma ser mais sensível que o Random Forest.


## No Gradient Boosting, o que significa dizer que cada árvore aprende os resíduos da anterior?
- resíduo = valor real − valor previsto
No Gradient Boosting, dizer que cada árvore aprende os resíduos significa que cada nova árvore é treinada para corrigir o erro da anterior. Primeiro o modelo faz uma previsão inicial, depois calcula o erro, que é a diferença entre o valor real e o previsto.

A próxima árvore é treinada para aprender esse erro, e a previsão final passa a ser a soma das árvores. Esse processo se repete várias vezes, fazendo com que o modelo vá minimizando a função de custo gradualmente, aprendendo padrões cada vez mais complexos.

Por isso o Gradient Boosting costuma ter alta performance, mas também pode sofrer overfitting se não controlar os hiperparâmetros.


## O que é Out-of-Bag no Random Forest e para que ele serve?
No Random Forest é usada a técnica de bagging, com bootstrap, onde cada árvore é treinada com uma amostra aleatória dos dados com reposição.

Como a amostragem é feita com reposição, alguns dados acabam não sendo selecionados para treinar uma determinada árvore. Esses dados que ficam de fora são chamados de Out-of-Bag.

Em média, cerca de um terço dos dados não é usado no treino de cada árvore, e esses dados podem ser usados para fazer uma validação interna do modelo. Como essas amostras não foram vistas pela árvore, a gente consegue usar elas para estimar o erro de generalização sem precisar de um conjunto de validação separado.

O Out-of-Bag não reduz diretamente o overfitting, mas ajuda a medir se o modelo está generalizando bem, sem precisar usar o conjunto de teste.


## Como o Gradient Boosting constrói as árvores sequencialmente e o que significa dizer que ele aprende os resíduos?
No Gradient Boosting, as árvores são construídas de forma sequencial, diferente do Random Forest, onde as árvores são independentes.

Primeiro o modelo constrói uma árvore inicial, faz a previsão e calcula o erro, que são os resíduos, ou seja, a diferença entre o valor real e o valor previsto. A próxima árvore é treinada para aprender esses resíduos, tentando corrigir os erros da árvore anterior.

Esse processo se repete várias vezes, e a previsão final é a soma das previsões de todas as árvores, o que faz com que o erro vá sendo minimizado gradualmente. O nome Gradient Boosting vem porque essa correção dos erros é feita usando o gradiente da função de custo, ou seja, o modelo vai na direção que mais reduz o erro.

Hiperparâmetros como learning_rate, n_estimators e max_depth são importantes para controlar o overfitting, porque muitas árvores ou árvores muito profundas podem fazer o modelo aprender demais os dados de treino.


## Por que Gradient Boosting precisa de learning rate e como ele influencia o overfitting?
No Gradient Boosting, as árvores são construídas de forma sequencial, onde cada nova árvore vem para corrigir os erros da anterior.

O learning rate controla o quanto cada nova árvore vai corrigir do erro anterior.

Se o learning rate for muito alto, cada árvore corrige uma parte muito grande do erro, o que pode fazer o modelo aprender muito rápido e acabar sofrendo overfitting.

Se o learning rate for menor, a correção é feita de forma mais gradual, então o modelo precisa de mais árvores, mas tende a generalizar melhor.

Por isso existe um trade-off entre learning_rate e n_estimators, onde learning rate menor costuma precisar de mais árvores, mas reduz o risco de overfitting.


## Por que Random Forest reduz overfitting em relação a uma única árvore?
O Random Forest reduz overfitting porque utiliza Bagging, ou seja, constrói várias árvores em paralelo usando amostragem com reposição (bootstrap) dos dados de treino.

Além disso, em cada divisão da árvore ele usa apenas um subconjunto aleatório das features, o que reduz a correlação entre as árvores.

Como resultado, o modelo final faz uma média das previsões, reduzindo a variância e tornando o modelo mais robusto que uma única árvore, que tende a overfitar facilmente.

O custo disso é maior custo computacional e menor interpretabilidade.


## Qual a principal diferença entre Random Forest e Gradient Boosting?
A principal diferença entre Random Forest e Gradient Boosting está na forma de construção do ensemble.

O Random Forest usa Bagging, onde várias árvores são treinadas de forma independente usando bootstrap dos dados, e o resultado final é a média das previsões. Isso reduz a variância e torna o modelo mais robusto.

Já o Gradient Boosting constrói as árvores de forma sequencial, onde cada nova árvore tenta corrigir o erro da anterior, reduzindo o bias do modelo.

Como consequência, o Random Forest costuma ser mais estável e menos sensível a overfitting, enquanto o Boosting costuma ter melhor performance, mas exige mais tuning e pode overfitar com mais facilidade.


## Por que Gradient Boosting pode overfitar mais que Random Forest?
O Gradient Boosting pode overfitar mais que Random Forest porque ele constrói as árvores de forma sequencial, onde cada nova árvore tenta corrigir o erro da anterior.

Isso faz com que o modelo reduza muito o bias, mas pode aumentar a variância, principalmente se o número de árvores for grande ou se as árvores forem muito profundas.

Além disso, o Boosting tem menos aleatoriedade que o Random Forest, então ele pode se ajustar demais aos dados de treino.

Por isso é importante controlar hiperparâmetros como learning_rate, n_estimators, max_depth e subsample para evitar overfitting.

## Qual a função do learning rate no Gradient Boosting?
- learning rate alto → risco de overfitting
- learning rate baixo → modelo mais estável

O learning rate no Gradient Boosting controla o quanto cada árvore contribui para a correção do erro da árvore anterior.

- Learning rate baixo faz com que cada árvore faça pequenas correções, o que torna o modelo mais estável, mas exige um número maior de árvores.
- Learning rate alto faz com que cada árvore corrija muito do erro anterior, o que pode fazer o modelo convergir mais rápido, mas aumenta o risco de overfitting.

Por isso o learning rate funciona como uma forma de regularização no Gradient Boosting, controlando o trade-off entre bias e variância.

## Por que Random Forest reduz variância mas não reduz muito bias?
Random Forest reduz variância porque combina várias árvores treinadas com bootstrap e subconjunto de features, e a média das previsões torna o modelo mais estável.

Porém, ele não reduz muito o bias porque cada árvore individual ainda tem a mesma capacidade de modelagem, então o erro sistemático do modelo não muda muito.

O ensemble por bagging reduz principalmente variância, enquanto técnicas como boosting conseguem reduzir bias.


## Por que Gradient Boosting consegue reduzir bias, enquanto Random Forest reduz mais variância?
O Gradient Boosting reduz bias porque constrói o modelo de forma sequencial, onde cada nova árvore é treinada para corrigir os erros da anterior, aumentando gradualmente a complexidade do modelo e melhorando o ajuste aos dados.

Já o Random Forest reduz principalmente a variância, pois treina várias árvores independentes em amostras diferentes usando bagging e depois faz a média das previsões. Como as árvores do Random Forest normalmente já têm baixo bias, o ensemble reduz a variância, mas não altera muito o bias.