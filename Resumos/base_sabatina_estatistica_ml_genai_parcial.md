# Base de Sabatina — Estatística, Machine Learning e GenAI

> **Material parcial:** esta cópia contém apenas o trecho recuperado da conversa original, até parte da seção de regressão.

Este material reúne conceitos que podem aparecer em uma sabatina de Ciência de Dados. A ideia é conseguir responder em três níveis: primeiro a **intuição**, depois a **explicação técnica** e, quando necessário, a **matemática por trás**.

---

## 1. Estatística e Base Matemática

### O que é uma função?

Uma função é uma regra que associa cada elemento de um conjunto de entrada a **exatamente um elemento de saída**.

Podemos representar uma função como:

```math
f(x)=y
```

O valor $`x`$ é a entrada e $`f(x)`$ é a saída produzida pela função.

Por exemplo:

```math
f(x)=2x+1
```

Se:

```math
x=3
```

então:

```math
f(3)=2(3)+1=7
```

Em Machine Learning, praticamente todo modelo pode ser interpretado como uma função:

```math
\hat y=f(X)
```

Recebemos determinadas características $`X`$ e o modelo produz uma previsão $`\hat y`$.

Por exemplo, em um modelo de crédito:

```math
f(\text{renda},\text{idade},\text{dívida},\text{histórico})
\rightarrow
P(\text{inadimplência})
```

Portanto, um modelo de Machine Learning é essencialmente uma função aprendida a partir dos dados.

#### Resposta para sabatina

Uma função é uma regra que associa cada entrada do domínio a uma única saída. Em Machine Learning, podemos enxergar o modelo como uma função que recebe as features $`X`$ e produz uma previsão $`\hat y`$.

---

### Domínio e imagem de uma função

Considere:

```math
f(x)=x^2
```

O **domínio** representa os valores que podem entrar na função.

Se estivermos trabalhando com números reais:

```math
Dom(f)=\mathbb{R}
```

porque qualquer número real pode ser elevado ao quadrado.

A **imagem** representa os valores que efetivamente podem sair da função.

Como:

```math
x^2\geq0
```

temos:

```math
Im(f)=[0,+\infty)
```

Existe ainda o conceito de **contradomínio**, que é o conjunto definido como possíveis saídas da função. Imagem e contradomínio não precisam ser iguais.

Em Machine Learning isso possui uma interpretação interessante. Em uma regressão logística:

```math
p(x)=\frac{1}{1+e^{-z}}
```

a saída sempre pertence ao intervalo:

```math
0<p(x)<1
```

Portanto, a imagem da função logística está entre 0 e 1, motivo pelo qual ela é adequada para representar probabilidades.

---

### Um menino tirou nota 1 e nota 10. A nota 5 seria um outlier?

Não podemos concluir isso apenas com essas informações.

Um **outlier não é simplesmente um valor diferente ou distante de outro valor**. Outlier é uma observação considerada atípica quando comparada à distribuição dos dados.

Se tivermos apenas:

```math
1,\quad10
```

não temos informação suficiente para entender a distribuição.

Além disso, se 5 nem sequer estiver presente no conjunto de dados, tecnicamente não podemos chamá-lo de outlier daquele conjunto.

Mesmo se tivéssemos:

```math
1,\quad5,\quad10
```

não poderíamos afirmar automaticamente que algum desses valores é outlier.

Precisaríamos conhecer a população ou uma quantidade maior de observações.

Por exemplo:

```math
5,\ 5,\ 6,\ 5,\ 6,\ 5,\ 40
```

Nesse conjunto, 40 parece muito mais claramente uma observação atípica.

Outliers podem ser avaliados por métodos como IQR, boxplot, z-score, distância, métodos robustos ou conhecimento de negócio.

#### Pegadinha

**Valor máximo ou mínimo não significa necessariamente outlier.**

Todo dataset possui um mínimo e um máximo. Isso não significa que sejam valores anormais.

#### Resposta para sabatina

Eu não poderia afirmar que 5 é um outlier apenas sabendo que existem notas 1 e 10. Um outlier é definido em relação à distribuição dos dados e ao contexto, e não apenas por estar distante de outro ponto.

---

## 2. O que é um modelo?

Um modelo é uma representação simplificada de alguma relação existente nos dados.

Em Machine Learning, queremos aprender uma função:

```math
f(X)=\hat y
```

que consiga aproximar uma relação desconhecida entre as variáveis de entrada e uma variável de interesse.

Podemos imaginar que exista uma relação verdadeira:

```math
Y=f(X)+\epsilon
```

onde:

- $`f(X)`$ representa o padrão existente;
- $`\epsilon`$ representa ruído ou fatores não observados;
- o nosso modelo tenta estimar $`f(X)`$.

Por isso escrevemos:

```math
\hat f(X)
```

O objetivo não é memorizar perfeitamente os dados disponíveis, mas encontrar uma aproximação que **generalize para dados novos**.

---

### Qual seria o modelo mais simples possível?

Para regressão, um dos modelos mais simples possíveis é simplesmente prever a média de $`Y`$.

Por exemplo, temos:

```math
Y=[100,200,300]
```

A média é:

```math
\bar y=200
```

Nosso modelo seria:

```math
\hat y=200
```

para qualquer observação.

Esse modelo ignora completamente as features.

Ele funciona como um **baseline**.

Inclusive, existe uma propriedade matemática importante: entre todas as previsões constantes possíveis, a média é aquela que minimiza a soma dos erros quadráticos.

Ou seja:

```math
\bar y =
\arg\min_c
\sum_i(y_i-c)^2
```

Por isso a média possui uma relação direta com o MSE.

Para classificação, um baseline extremamente simples poderia ser prever sempre a classe majoritária ou utilizar a proporção observada de positivos como probabilidade constante.

#### Sabatina

Se me perguntassem qual é o modelo de regressão mais simples, eu poderia usar a média do target. Ele ignora todas as features e prevê sempre o mesmo valor, funcionando como baseline para verificar se modelos mais sofisticados realmente aprenderam alguma relação útil.

---

## 3. Classificação

### Explicar KNN

KNN significa **K-Nearest Neighbors**, ou K vizinhos mais próximos.

É um algoritmo supervisionado que pode ser usado tanto para classificação quanto para regressão.

Sua característica principal é que ele praticamente não aprende uma equação explícita durante o treinamento.

Quando chega uma nova observação, o algoritmo procura os $`K`$ exemplos de treinamento mais próximos.

Suponha:

```math
K=5
```

Os cinco vizinhos encontrados possuem classes:

```math
1,\ 1,\ 0,\ 1,\ 0
```

Temos três observações da classe 1 e duas da classe 0.

O KNN classificaria o novo ponto como:

```math
\hat y=1
```

Na regressão, em vez de votação, normalmente utilizamos a média dos valores dos vizinhos.

---

### Como o KNN determina quem está próximo?

Normalmente por uma medida de distância.

A mais conhecida é a distância euclidiana:

```math
d(x,z)=
\sqrt{
\sum_{j=1}^{p}(x_j-z_j)^2
}
```

Por isso a escala das variáveis importa muito.

Imagine:

- idade: 18–80;
- salário: 1.000–100.000.

Sem padronização, a variável salário pode dominar completamente a distância.

Por isso KNN normalmente exige normalização ou padronização das features.

---

### O valor de K

Um $`K`$ muito pequeno produz um modelo bastante flexível.

Com:

```math
K=1
```

cada observação é basicamente classificada de acordo com seu vizinho mais próximo.

Isso pode produzir:

- baixa tendência de viés;
- alta variância;
- maior sensibilidade ao ruído;
- maior risco de overfitting.

Um $`K`$ maior suaviza a fronteira de decisão.

Temos então:

- maior viés;
- menor variância.

Essa é uma aplicação clássica do trade-off entre viés e variância.

---

### KNN e plano linear

Uma pegadinha importante é perguntar:

**A fronteira de decisão do KNN é linear?**

Normalmente, não.

Imagine duas features:

```math
X_1
```

e:

```math
X_2
```

que formam um plano bidimensional.

Cada ponto de treinamento ocupa uma posição nesse plano.

O KNN classifica cada região com base nos vizinhos mais próximos.

Como consequência, sua fronteira pode assumir um formato extremamente irregular.

Portanto, diferente de uma regressão logística simples, KNN **não precisa construir uma reta ou hiperplano linear**.

A fronteira é determinada pela geometria local dos dados.

---

### Regressão logística

Apesar do nome "regressão", a regressão logística é utilizada principalmente para **classificação**.

Em uma classificação binária queremos estimar:

```math
P(Y=1|X)
```

Começamos com uma combinação linear:

```math
z=\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_px_p
```

O problema é que $`z`$ pode assumir qualquer número:

```math
-\infty < z < +\infty
```

e uma probabilidade precisa estar entre:

```math
0\leq P\leq1
```

Por isso aplicamos a função logística ou sigmoide:

```math
p=
\frac{1}{1+e^{-z}}
```

Agora temos:

```math
0<p<1
```

Por exemplo:

```math
P(Y=1|X)=0.82
```

Com um threshold de 0,5:

```math
0.82 > 0.5
```

classificamos a observação como classe 1.

---

### Odds e logit

A regressão logística pode ser escrita em termos de odds:

```math
Odds=\frac{p}{1-p}
```

Aplicando log:

```math
\log\left(\frac{p}{1-p}\right)
=
\beta_0+\beta_1X_1+\cdots+\beta_pX_p
```

Esse valor é chamado de **logit**.

Portanto, a regressão logística é linear no **log-odds**, e não diretamente na probabilidade.

---

### Por que não utilizar simplesmente 1 − probabilidade?

Na classificação binária podemos sim utilizar:

```math
P(Y=0|X)=1-P(Y=1|X)
```

Se:

```math
P(Y=1)=0.8
```

automaticamente:

```math
P(Y=0)=0.2
```

Não precisamos construir outro modelo para calcular a probabilidade da classe 0.

Se a pergunta for "por que não usar $`1-p`$?", a resposta é que **podemos usar**: $`1-p`$ é exatamente a probabilidade complementar da outra classe em classificação binária.

Porém isso não substitui a necessidade de transformar o score linear em probabilidade. A sigmoide é que garante uma saída entre 0 e 1.

Uma pegadinha seria confundir:

```math
1-p
```

com uma alternativa à função sigmoide.

Não é.

Primeiro calculamos:

```math
p=\sigma(z)
```

Depois podemos calcular:

```math
P(Y=0)=1-p
```

---

### Regressão logística versus regressão linear

A regressão linear modela diretamente uma variável contínua:

```math
\hat y=
\beta_0+\beta_1X_1+\cdots+\beta_pX_p
```

Sua saída pode assumir qualquer valor real.

A regressão logística utiliza uma combinação linear semelhante:

```math
z=
\beta_0+\beta_1X_1+\cdots+\beta_pX_p
```

mas passa esse resultado pela sigmoide:

```math
P(Y=1|X)=\sigma(z)
```

A diferença fundamental está no problema sendo resolvido.

Na regressão linear, normalmente temos target contínuo e podemos estimar os parâmetros minimizando erros quadráticos.

Na regressão logística, o target é categórico binário e normalmente estimamos os parâmetros por máxima verossimilhança, equivalente à minimização da log loss ou binary cross-entropy.

---

### Precision e Recall

Considere uma classificação binária.

Temos:

- TP: positivo verdadeiro;
- FP: falso positivo;
- TN: negativo verdadeiro;
- FN: falso negativo.

A precision responde:

> Das observações que o modelo classificou como positivas, quantas realmente eram positivas?

```math
Precision=
\frac{TP}{TP+FP}
```

O recall responde:

> Das observações realmente positivas, quantas o modelo conseguiu encontrar?

```math
Recall=
\frac{TP}{TP+FN}
```

---

### Precision e Recall em um plano X₁ e X₂

Imagine duas features:

```math
X_1
```

no eixo horizontal e:

```math
X_2
```

no eixo vertical.

Cada observação ocupa algum ponto no plano.

O modelo cria uma **fronteira de decisão**.

De um lado da fronteira:

```math
\hat y=0
```

e do outro:

```math
\hat y=1
```

Agora precisamos comparar essa classificação com a classe verdadeira.

Um ponto realmente positivo colocado na região positiva é:

```math
TP
```

Um ponto realmente negativo que caiu na região positiva é:

```math
FP
```

Um ponto positivo que caiu na região negativa é:

```math
FN
```

Um ponto negativo corretamente localizado na região negativa é:

```math
TN
```

Precision e recall não são propriedades geométricas dos eixos $`X_1`$ e $`X_2`$. Elas surgem quando comparamos **a região prevista pelo modelo com os labels verdadeiros**.

Mudando o threshold, alteramos quais observações são consideradas positivas.

Um threshold menor tende a classificar mais observações como positivas:

- recall tende a subir;
- falsos positivos podem aumentar;
- precision pode cair.

Um threshold maior tende a ser mais conservador:

- menos positivos previstos;
- precision pode aumentar;
- recall pode cair.

---

## 4. Árvores

### Como outliers afetam árvores?

Árvores de decisão tendem a ser menos sensíveis a outliers nas features do que modelos baseados diretamente em distâncias ou regressões lineares.

Isso acontece porque a árvore trabalha principalmente com cortes:

```math
X_j < c
```

ou:

```math
X_j \geq c
```

Imagine:

```math
idade=[20,25,30,35,500]
```

Uma regressão linear pode ser bastante influenciada pelo valor 500.

Uma árvore pode simplesmente produzir um corte que isola aquela região.

Mas dizer que "árvores são imunes a outliers" estaria errado.

Em árvores de regressão, um valor extremo do target pode alterar bastante a redução de erro utilizada para escolher os splits, principalmente se o critério estiver baseado em erro quadrático.

Outliers também podem causar:

- folhas específicas para poucos pontos;
- splits desnecessários;
- maior complexidade;
- overfitting.

Portanto, árvores são geralmente **mais robustas**, mas não completamente imunes.

---

### Gradient Boosting

Gradient Boosting constrói vários modelos sequencialmente.

A ideia é:

```math
F_0(x)
```

produz uma previsão inicial.

Depois adicionamos outro modelo:

```math
F_1(x)=F_0(x)+\eta h_1(x)
```

Depois:

```math
F_2(x)=F_1(x)+\eta h_2(x)
```

e assim sucessivamente.

Temos:

```math
F_M(x)
=
F_0(x)+
\eta\sum_{m=1}^{M}h_m(x)
```

onde:

- $`h_m(x)`$ normalmente é uma árvore;
- $`\eta`$ é o learning rate;
- $`M`$ é o número de árvores.

O grande diferencial é que cada nova árvore tenta corrigir os erros do ensemble atual.

---

### Gradiente em relação a quê?

Essa é uma pergunta excelente de sabatina.

No Gradient Boosting, calculamos a derivada da função de perda **em relação à previsão atual do modelo**.

Se nossa função de perda for:

```math
L(y,F(x))
```

calculamos:

```math
\frac{\partial L(y,F(x))}
{\partial F(x)}
```

Mais especificamente, usamos o **gradiente negativo**:

```math
r_i=
-
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
```

Esses valores funcionam como pseudo-resíduos.

A próxima árvore tenta aproximar esses pseudo-resíduos.

Portanto, uma resposta importante é:

> No Gradient Boosting, o gradiente utilizado no processo de boosting é calculado em relação à predição ou score atual do modelo, e indica a direção em que precisamos alterar as previsões para reduzir a função de perda.

No XGBoost, a otimização vai além: ele utiliza tanto primeira quanto segunda derivadas da loss:

```math
g_i=
\frac{\partial L}{\partial \hat y_i}
```

e:

```math
h_i=
\frac{\partial^2 L}
{\partial \hat y_i^2}
```

Esses são chamados de gradient e Hessian.

---

### O que acontece se adicionarmos infinitamente mais árvores?

#### Random Forest

Random Forest cria árvores **independentes ou paralelas** e depois agrega os resultados.

Classificação:

```math
\text{votação}
```

Regressão:

```math
\text{média}
```

Cada árvore é construída com aleatoriedade nos dados e/ou nas features.

Quando aumentamos muito o número de árvores, o erro tende a **estabilizar**.

Adicionar mais árvores normalmente não causa o mesmo tipo de overfitting relacionado ao número de iterações observado em boosting.

Com muitas árvores:

```math
\operatorname{Var}\;\downarrow
```

até um determinado limite.

O custo computacional continua aumentando, mas o ganho adicional tende a ficar cada vez menor.

Isso não significa que um Random Forest seja incapaz de overfitting. Profundidade, dados ruidosos, features e hiperparâmetros continuam importantes. A diferença é que aumentar apenas o número de árvores normalmente não faz o erro de generalização explodir.

---

#### XGBoost

No XGBoost a história é diferente.

Cada nova árvore tenta corrigir o ensemble existente:

```math
F_m(x)=F_{m-1}(x)+\eta h_m(x)
```

Se continuarmos adicionando árvores indefinidamente, o modelo pode começar a aprender:

- ruído;
- particularidades do treinamento;
- observações anômalas.

Consequentemente podemos ter overfitting.

Por isso usamos mecanismos como:

- learning rate;
- `max_depth`;
- `min_child_weight`;
- `subsample`;
- `colsample`;
- regularização L1/L2;
- early stopping.

Uma forma resumida de lembrar é:

**Random Forest:** mais árvores → geralmente estabilização.

**Boosting:** mais árvores → pode continuar reduzindo treino e eventualmente piorar teste.

---

## 5. Regressão

### Explicar regressão linear

A regressão linear procura modelar uma relação entre uma variável resposta contínua e uma ou mais variáveis explicativas.

Na regressão simples:

```math
Y=\beta_0+\beta_1X+\epsilon
```

Na regressão múltipla:

```math
Y=
\beta_0+
\beta_1X_1+
\beta_2X_2+
\cdots+
\beta_pX_p+
\epsilon
```

O modelo estimado é:

```math
\hat Y=
\hat\beta_0+
\hat\beta_1X_1+
\cdots+
\hat\beta_pX_p
```

Cada coeficiente representa a alteração esperada em $`Y`$ associada à alteração de uma unidade naquela variável, mantendo as demais constantes.

Por exemplo:

```math
\hat Y=100+2X
```

significa que um aumento de uma unidade em $`X`$ está associado a um aumento médio de 2 unidades em $`Y`$.

---

### Quais são os dois métodos para encontrar os coeficientes?

Uma resposta comum é:

1. solução analítica por Mínimos Quadrados Ordinários;
2. solução numérica por Gradient Descent.

É importante entender que estamos tentando resolver o **mesmo problema de otimização** por formas diferentes.

---

### Mínimos Quadrados Ordinários — MQO

O MQO procura os coeficientes que minimizam:

```math
RSS=
\sum_{i=1}^{n}(y_i-\hat y_i)^2
```

ou seja, a soma dos quadrados dos resíduos.

O resíduo é:

```math
e_i=y_i-\hat y_i
```

Elevamos ao quadrado para evitar que erros positivos e negativos simplesmente se cancelem.

---

### Gradient Descent

Outra opção é definir uma função de custo, por exemplo:

```math
J(\beta)
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
```

e atualizar os parâmetros iterativamente:

```math
\beta_j
\leftarrow
\beta_j
-
\eta
\frac{\partial J}
{\partial \beta_j}
```

onde:

```math
\eta
```

é o learning rate.

A derivada indica como a loss varia quando alteramos o parâmetro.

Descemos na direção oposta ao gradiente porque queremos minimizar a função.

---

### Como fazer MQO "na mão"?

Em forma matricial:

```math
Y=X\beta+\epsilon
```

Queremos minimizar:

```math
(Y-X\beta)^T(Y-X\beta)
```

Derivando em relação a $`\beta`$ e igualando a zero chegamos às equações normais:

```math
X^TX\hat\beta=X^TY
```

Se $`X^TX`$ for invertível:

```math
\boxed{
\hat\beta=
(X^TX)^{-1}X^TY
}
```

Essa é a solução fechada clássica do MQO.

Em aplicações reais, bibliotecas numéricas normalmente não calculam literalmente a inversa dessa matriz, porque existem métodos numericamente mais estáveis, como decomposição QR ou SVD.

Mas conceitualmente essa fórmula é extremamente importante para a sabatina.

---

### O que significa a matriz ser invertível?

Na solução clássica:

```math
\hat\beta=
(X^TX)^{-1}X^TY
```

precisamos que:

```math
X^TX
```

tenha inversa.

Isso está relacionado a $`X`$ possuir **posto completo nas colunas**.

Uma consequência importante é que não pode existir multicolinearidade perfeita entre as features.

Imagine:

```math
X_3=X_1+X_2
```

Agora uma coluna pode ser perfeitamente construída a partir de outras.

Não temos informação independente suficiente para determinar uma única combinação de coeficientes.

Consequentemente:

```math
X^TX
```

se torna singular.

A solução clássica usando inversa deixa de existir de forma única.

Podemos tratar isso por:

- remover variáveis redundantes;
- utilizar pseudoinversa;
- regularização;
- métodos numéricos apropriados.

#### Pegadinha

Correlação alta não significa necessariamente que a matriz seja não invertível.

O problema matemático extremo é a **dependência linear exata**.

---

### Outlier e regressão linear

A regressão linear utilizando MQO pode ser bastante sensível a outliers.

Isso acontece porque minimizamos:

```math
\sum e_i^2
```

Imagine dois resíduos:

```math
e_1=2
```

e:

```math
e_2=20
```

Depois de elevar ao quadrado:

```math
2^2=4
```

```math
20^2=400
```

O segundo erro passa a ter um peso enorme na função de custo.

Consequentemente, algumas observações extremas podem puxar bastante a reta de regressão.

---

> **Nota sobre esta cópia:** o acesso à mensagem original foi limitado a 20.000 caracteres. O texto acima preserva o trecho recuperado; o restante do material não está incluído neste arquivo.

