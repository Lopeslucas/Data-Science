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
# Perguntas e respostas:

## Como a árvore de decisão escolhe o melhor split? (critério de divisão)
A árvore de decisão escolhe o melhor split avaliando todas as possíveis divisões das variáveis e selecionando aquela que maximiza a separação dos dados naquele nó.

Para problemas de classificação, os critérios mais usados são Gini impurity e Entropy, que medem o grau de impureza das classes. O split escolhido é aquele que reduz mais a impureza, ou seja, que gera maior information gain.

Para regressão, normalmente usamos redução de variância ou MSE para escolher o split.

A árvore é considerada um algoritmo guloso porque ela escolhe sempre a melhor divisão local em cada nó, sem garantir que isso leve à melhor solução global.

## Por que árvores de decisão tendem a overfitting? E como Random Forest reduz isso?
Árvores de decisão tendem a overfitting porque são modelos de baixa bias e alta variância.
Elas podem crescer até separar completamente os dados de treino, aprendendo inclusive o ruído, o que reduz a capacidade de generalização.

O Random Forest reduz esse problema usando bagging, onde várias árvores são treinadas em amostras bootstrap diferentes do dataset, e a previsão final é feita pela média ou votação. Além disso, em cada split o Random Forest usa apenas um subconjunto aleatório de features, o que reduz a correlação entre as árvores.

Essa combinação reduz a variância do modelo e diminui o risco de overfitting, tornando o Random Forest mais robusto que uma única árvore.

## Por que Random Forest costuma overfitar menos que uma árvore de decisão?
O Random Forest costuma sofrer menos overfitting porque ele treina várias árvores independentes usando amostras diferentes dos dados, através de bootstrap, e depois faz a predição pela média ou votação.

Além disso, em cada divisão ele usa apenas um subconjunto aleatório de features, o que reduz a correlação entre as árvores. Isso faz com que o modelo tenha menor variância e fique mais robusto, reduzindo o risco de overfitting quando comparado com uma única árvore de decisão, que tende a se ajustar muito aos dados de treino.

Em contrapartida, o Random Forest tem um custo computacional maior e é menos interpretável do que uma árvore simples.

## O que acontece se eu aumentar muito a profundidade de uma árvore? E como controlar isso?
Modelos baseados em árvore, quando têm profundidade muito alta, tendem a sofrer overfitting, porque a árvore continua fazendo divisões até separar quase perfeitamente os dados de treino, e com isso ela acaba capturando o ruído ao invés de aprender só o padrão.

Para controlar isso, a gente pode limitar a profundidade da árvore usando max_depth, definir um número mínimo de amostras para dividir com min_samples_split ou min_samples_leaf, e também limitar o número de folhas.

Essas técnicas ajudam a reduzir a variância do modelo e evitam que ele fique muito ajustado aos dados de treino.
        
## O que é learning rate no Gradient Boosting e o que acontece se ele for muito alto ou muito baixo?
O learning rate no Gradient Boosting controla o quanto cada nova árvore influencia no modelo final.

Como o boosting cria árvores sequenciais, cada árvore corrige o erro da anterior, e o learning rate define o tamanho desse ajuste. Se o learning rate for muito alto, o modelo aprende rápido demais e pode sofrer overfitting.

Se for muito baixo, o modelo aprende mais devagar e precisa de mais árvores, mas geralmente generaliza melhor. Por isso, o learning rate é um hiperparâmetro importante para controlar o equilíbrio entre bias e variância.

## Qual a diferença entre Bagging e Boosting?
O bagging é um método onde a gente treina várias árvores independentes usando amostras diferentes dos dados, geralmente com bootstrap, e depois faz a predição pela média ou votação. Isso ajuda a reduzir a variância e deixa o modelo mais estável.

Já o boosting funciona de forma sequencial, onde cada nova árvore é treinada para corrigir o erro da anterior. Com isso o modelo vai melhorando aos poucos, reduzindo o viés e conseguindo aprender padrões mais complexos.

Um exemplo de bagging é o Random Forest, e de boosting são modelos como Gradient Boosting, XGBoost e LightGBM.