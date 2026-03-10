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