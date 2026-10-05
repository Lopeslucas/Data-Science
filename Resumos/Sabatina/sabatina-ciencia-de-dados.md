# Sabatina de Ciência de Dados — 67 perguntas e respostas

Material de revisão técnica em PT-BR, elaborado a partir das perguntas e correções do treino “Treino de sabatina”. As 67 perguntas seguem a ordem original; as respostas consolidam os conceitos em linguagem direta, adequada a uma entrevista técnica.

Cada resposta inclui um exemplo para apoiar a revisão. Os números e cenários adicionais são ilustrativos, salvo quando reproduzem o enunciado; não representam resultados reais de modelos.

## Pergunta 1

Qual é a diferença entre um modelo supervisionado e um modelo não supervisionado?

### Resposta de sabatina

No aprendizado supervisionado, nós temos uma variável-alvo conhecida durante o treinamento. O modelo aprende uma relação entre as variáveis de entrada, \(X\), e esse alvo, \(y\). Dependendo do problema, podemos ter classificação, quando queremos prever uma classe, ou regressão, quando queremos prever um valor numérico.

Já no aprendizado não supervisionado, nós não temos uma variável-alvo conhecida. O algoritmo tenta encontrar padrões ou estruturas nos próprios dados. Um exemplo é a clusterização, em que podemos agrupar observações semelhantes.

### Exemplo prático

Compare duas tarefas com a mesma base de clientes:

- **Supervisionado:** usar renda e histórico de compras para prever churn, com exemplos históricos rotulados como “saiu” ou “permaneceu”.
- **Não supervisionado:** usar essas características para descobrir segmentos, sem fornecer rótulos de grupos ao algoritmo.

## Pergunta 2

Qual é a diferença entre um modelo de classificação e um modelo de regressão?

### Resposta de sabatina

Um modelo de classificação é usado quando queremos prever uma classe ou categoria. Essa classificação pode ser binária, como churn ou não churn, ou multiclasse, como baixo, médio ou alto risco.

Já um modelo de regressão é utilizado quando queremos prever uma quantidade numérica, geralmente contínua, como o valor de um imóvel, a renda de um cliente ou o custo de uma conta de energia.

### Exemplo prático

Em uma seguradora:

| Tarefa | Saída ilustrativa | Tipo |
|---|---|---|
| Prever se ocorrerá um sinistro | Sim ou não | Classificação |
| Prever o valor de um sinistro | R$ 8.500 | Regressão |

Uma classe codificada como 0 ou 1 continua sendo uma categoria, mesmo sendo representada por um número.

## Pergunta 3

Como funciona uma regressão linear?

### Resposta de sabatina

A regressão linear busca modelar a relação entre uma variável-alvo numérica e uma ou mais variáveis explicativas. Ela representa essa relação por uma equação linear, com um intercepto e coeficientes associados às variáveis.

No treinamento, o modelo estima esses coeficientes de forma a encontrar a reta, ou o hiperplano, que melhor se ajusta aos dados. No caso clássico de mínimos quadrados, isso é feito minimizando a soma dos quadrados dos resíduos, que são as diferenças entre os valores reais e os valores previstos.

### Exemplo prático

Considere o modelo ilustrativo `renda prevista = 2.000 + 300 × anos de experiência`.

1. Para 5 anos, a previsão é `2.000 + 300 × 5 = R$ 3.500`.
2. Se a renda real for R$ 3.800, o resíduo é `3.800 − 3.500 = R$ 300`.
3. A contribuição desse caso para a soma dos quadrados dos resíduos é `300² = 90.000` reais².

## Pergunta 4

Quais considerações e cuidados precisamos ter ao treinar uma regressão linear?

### Resposta de sabatina

Eu verificaria se a média do alvo pode ser descrita por uma combinação linear dos preditores, inclusive após transformações, e avaliaria multicolinearidade, valores ausentes, pontos influentes e vazamento de informação. Separaria treino e teste de acordo com o problema e analisaria os resíduos e o desempenho fora da amostra.

Para a interpretação estatística, é importante que o erro tenha média zero condicionada aos preditores. Dependência entre erros e heterocedasticidade exigem cuidados com os erros-padrão; normalidade dos erros é relevante para testes e intervalos clássicos exatos em pequenas amostras, mas não é requisito para calcular o ajuste por mínimos quadrados.

### Exemplo prático

Ao prever preços de imóveis, os resíduos podem revelar diferentes problemas:

| Padrão observado | Possível problema | Ação a avaliar |
|---|---|---|
| Curva nos resíduos versus previsões | Relação mal especificada | Transformações ou termos não lineares |
| Dispersão cresce com o preço previsto | Heterocedasticidade | Erros-padrão robustos para inferência |
| Um imóvel altera muito os coeficientes | Ponto influente | Verificar o dado e analisar sua influência |

Esses padrões orientam o diagnóstico; não provam sozinhos sua causa.

## Pergunta 5

Como calculamos ou estimamos os betas da regressão linear?

### Resposta de sabatina

Estimamos os betas por mínimos quadrados ordinários (MQO), minimizando a soma dos quadrados dos resíduos: `Σ(yᵢ − ŷᵢ)²`. Na regressão simples, `ŷ = β₀ + β₁x`: β₀ é o intercepto, e β₁ representa a variação prevista no alvo por unidade de x.

Na forma matricial, se X tem colunas linearmente independentes, a solução é `β̂ = (XᵀX)⁻¹Xᵀy`, incluindo uma coluna de uns para o intercepto. Na prática, decomposições como QR ou SVD são preferíveis a calcular explicitamente essa inversa.

### Exemplo prático

Para os pontos `(x, y) = (1, 3), (2, 5), (3, 7)`:

- Médias: `x̄ = 2` e `ȳ = 5`.
- Inclinação: `β̂₁ = Σ[(xᵢ − x̄)(yᵢ − ȳ)] / Σ(xᵢ − x̄)² = 4/2 = 2`.
- Intercepto: `β̂₀ = ȳ − β̂₁x̄ = 5 − 2 × 2 = 1`.
- Equação ajustada: `ŷ = 1 + 2x`, com resíduos zero nesse exemplo.

## Pergunta 6

Quais são as formas de chegar aos coeficientes da equação?

### Resposta de sabatina

Podemos resolver o problema de mínimos quadrados diretamente por álgebra linear, por exemplo com decomposições QR ou SVD, ou utilizar otimização iterativa, como gradiente descendente. No método iterativo, atualizamos os coeficientes na direção que reduz a função de custo até um critério de convergência. A escolha depende do tamanho dos dados, da estabilidade numérica e dos recursos disponíveis.

### Exemplo prático

Considere `ŷ = βx`, sem intercepto, com um único ponto `x = 1, y = 4` e perda `L = (4 − β)²/2`.

- **Solução direta:** o mínimo ocorre em `β = 4`.
- **Gradiente descendente:** `β_novo = β − η(β − 4)`.
- Começando em `β = 0`, com `η = 0,5`, as atualizações são `0 → 2 → 3 → 3,5 → … → 4`.

## Pergunta 7

Quais cuidados você deve tomar no data prep para uma regressão linear?

### Resposta de sabatina

No data prep de uma regressão linear, eu teria cuidado com valores ausentes, outliers e pontos influentes, porque eles podem afetar bastante os coeficientes. Também avaliaria multicolinearidade entre as variáveis explicativas, podendo utilizar correlação e VIF.

Variáveis categóricas precisam ser codificadas adequadamente, e também devemos evitar data leakage. A padronização não é obrigatória para uma regressão linear simples, mas se torna especialmente importante quando utilizamos regularização, como Ridge ou Lasso.

### Exemplo prático

Um fluxo para prever renda com Ridge:

1. Separar treino e teste.
2. Aprender a mediana para imputação apenas no treino.
3. Aprender a codificação das categorias e a padronização no treino.
4. Aplicar essas transformações à validação e ao teste.
5. Ajustar a regularização usando validação.

Na validação cruzada, repetir o aprendizado das transformações dentro de cada divisão, por meio de um pipeline, evita vazamento.

## Pergunta 8

Com quais tipos de variável a regressão linear trabalha bem?

### Resposta de sabatina

A regressão linear trabalha diretamente com variáveis representadas numericamente, tanto contínuas quanto discretas. Variáveis categóricas também podem ser utilizadas, mas precisam de uma codificação adequada. Por exemplo, uma variável nominal pode ser representada por dummies ou one-hot encoding, enquanto uma variável binária pode ser representada como zero e um.

### Exemplo prático

Possíveis preditores de renda:

| Variável original | Tipo | Representação possível |
|---|---|---|
| Idade | Quantitativa | 35 |
| Número de dependentes | Quantitativa discreta | 2 |
| Empregado | Categórica binária | 0 ou 1 |
| Região | Categórica nominal | Dummies por região |

Representar numericamente permite o cálculo, mas não garante que a relação com o alvo seja bem modelada por uma função linear.

## Pergunta 9

Existe algum tipo de variável que exige tratamento específico?

### Resposta de sabatina

Sim. Principalmente as variáveis categóricas, porque a regressão precisa de uma representação numérica. Para variáveis nominais, podemos utilizar dummies ou one-hot encoding. Para variáveis ordinais, podemos utilizar uma codificação que preserve a ordem das categorias. Também devemos ter cuidado para não criar uma ordem artificial e, no caso de target encoding, evitar data leakage. Uma codificação ordinal por inteiros também supõe um efeito linear por incremento do código; se isso não fizer sentido, podemos usar dummies.

### Exemplo prático

Para a variável “cor”, codificar `azul = 1`, `verde = 2` e `vermelho = 3` faria a regressão assumir incrementos iguais de efeito entre as cores.

- Com azul como referência, crie `é_verde` e `é_vermelho`.
- Azul: `(0, 0)`; verde: `(1, 0)`; vermelho: `(0, 1)`.
- Cada cor passa a ter seu próprio efeito em relação à referência.

## Pergunta 10

O que você quer dizer quando afirma que a regressão linear trabalha com variáveis numéricas?

### Resposta de sabatina

Quando digo que a regressão linear trabalha com variáveis numéricas, significa que as features precisam estar representadas numericamente para participarem da equação. O modelo multiplica cada variável pelo seu respectivo coeficiente e soma esses valores para produzir a previsão. Isso não significa que todas as variáveis precisam ser originalmente quantitativas: uma variável categórica também pode ser utilizada, desde que seja adequadamente codificada, por exemplo por meio de dummies.

### Exemplo prático

Uma previsão pode combinar números medidos e categorias codificadas:
`renda prevista = 1.500 + 100 × experiência + 400 × tem_superior`.

- Experiência: 5 anos.
- Ensino superior: sim, representado por 1.
- Previsão: `1.500 + 100 × 5 + 400 × 1 = R$ 2.400`.

## Pergunta 11

É possível utilizar uma variável categórica em uma regressão linear?

### Resposta de sabatina

Sim. Uma variável categórica pode ser utilizada em regressão linear desde que seja representada numericamente de forma adequada. Para variáveis nominais, por exemplo, podemos utilizar variáveis dummy ou one-hot encoding, evitando criar uma ordem artificial entre as categorias. Com intercepto e uma variável de C categorias, normalmente usamos C − 1 dummies e uma categoria de referência para evitar dependência linear perfeita.

### Exemplo prático

Para três regiões, com Sul como referência:

| Região | É Sudeste | É Nordeste |
|---|---:|---:|
| Sul | 0 | 0 |
| Sudeste | 1 | 0 |
| Nordeste | 0 | 1 |

Em `ŷ = 2.000 + 500 × é_Sudeste − 200 × é_Nordeste`, as previsões são R$ 2.000, R$ 2.500 e R$ 1.800, respectivamente, mantendo outros preditores constantes.

## Pergunta 12

Como você utilizaria, na mesma regressão, as variáveis “idade” e “segundos de vida”?

### Resposta de sabatina

Eu não utilizaria as duas sem necessidade, porque idade e segundos de vida representam praticamente a mesma informação e seriam altamente colineares. Isso pode gerar multicolinearidade e instabilidade nos coeficientes. Como essa redundância já é conhecida, eu manteria apenas uma das variáveis. Em outros cenários com muitas variáveis correlacionadas, também poderia avaliar técnicas de regularização, como Lasso.

### Exemplo prático

Suponha uma conversão exata simplificada: `segundos = 31.536.000 × idade_em_anos`.

- Uma coluna é múltipla da outra: existe colinearidade perfeita.
- Diferentes pares de coeficientes produzem a mesma previsão.
- Manter apenas idade elimina essa redundância.

Se idade estiver arredondada em anos, a equivalência pode não ser exata, mas a correlação continuará muito alta.

## Pergunta 13

Os coeficientes da regressão linear podem ser usados para interpretar o modelo?

### Resposta de sabatina

Sim. Os coeficientes permitem interpretar a relação entre cada variável explicativa e o alvo. Mantendo as demais variáveis constantes, o coeficiente indica quanto esperamos que Y varie quando aquela variável X aumenta uma unidade. O sinal mostra a direção da associação. Porém, não podemos simplesmente comparar o tamanho dos coeficientes para determinar qual variável é mais importante, principalmente quando as features possuem escalas diferentes. Essa interpretação descreve uma associação condicional e, por si só, não demonstra causalidade.

### Exemplo prático

No modelo `renda prevista = 2.000 + 300 × experiência + 500 × tem_superior`:

- Um ano adicional de experiência está associado a **R$ 300 a mais** de renda prevista, mantendo escolaridade constante.
- Ter ensino superior está associado a **R$ 500 a mais**, mantendo experiência constante.
- Isso não prova que obter o diploma cause esse aumento de renda.

## Pergunta 14

É possível afirmar que uma variável com coeficiente maior é mais relevante que outra com coeficiente menor?

### Resposta de sabatina

Não. Não podemos afirmar que uma variável é mais relevante apenas porque possui um coeficiente maior, porque a magnitude do coeficiente depende da escala e da unidade de medida da variável. Por exemplo, representar renda em reais ou em milhares de reais altera o coeficiente sem alterar a informação da variável. Para tornar as magnitudes mais comparáveis, podemos padronizar as features, mas ainda devemos considerar outros fatores antes de interpretar importância.

### Exemplo prático

Compare duas escritas da mesma relação:

- `gasto previsto = 100 + 0,1 × renda_em_reais`.
- `gasto previsto = 100 + 100 × renda_em_milhares`.

Para renda de R$ 5.000, ambas preveem R$ 600. O coeficiente mudou de 0,1 para 100, mas a informação e a previsão são idênticas.

## Pergunta 15

Quais tipos de regularização você conhece?

### Resposta de sabatina

As principais são Lasso, Ridge e Elastic Net. O Lasso utiliza penalização L1, baseada no valor absoluto dos coeficientes, e pode zerar alguns deles, funcionando também como seleção de variáveis. O Ridge utiliza penalização L2, baseada no quadrado dos coeficientes, reduzindo sua magnitude sem normalmente zerá-los. Já o Elastic Net combina L1 e L2. A intensidade da regularização é controlada por um hiperparâmetro, normalmente chamado lambda ou alpha.

### Exemplo prático

Para coeficientes `β = (3, −4)`, desconsiderando o intercepto:

| Penalização | Cálculo | Valor |
|---|---|---:|
| L1 | `abs(3) + abs(−4)` | 7 |
| L2 | `3² + (−4)²` | 25 |

O Lasso adiciona à perda um múltiplo de L1; o Ridge, de L2. O Elastic Net combina as duas, com fatores de escala que dependem da convenção utilizada.

## Pergunta 16

Por que utilizar cada tipo de regularização?

### Resposta de sabatina

Eu utilizaria Lasso quando quero uma solução mais esparsa e seleção automática de variáveis, porque a penalização L1 pode zerar coeficientes. Utilizaria Ridge quando quero manter as variáveis, mas reduzir a magnitude e aumentar a estabilidade dos coeficientes, sendo bastante útil quando existem features correlacionadas. Já o Elastic Net combina os dois comportamentos e pode ser interessante quando temos muitas variáveis correlacionadas e também queremos seleção de features. De forma geral, a regularização ajuda a reduzir overfitting, adicionando um pouco de viés para reduzir a variância do modelo.

### Exemplo prático

Escolhas ilustrativas:

- **Lasso:** entre 200 variáveis, buscar uma solução com poucos coeficientes não nulos.
- **Ridge:** estabilizar coeficientes de medidas de gastos mensais muito correlacionadas.
- **Elastic Net:** combinar seleção de variáveis com maior estabilidade em grupos de variáveis correlacionadas.

Em todos os casos, escolher a intensidade da penalização por validação; nenhum resultado específico é garantido só pela escolha do método.

## Pergunta 17

Em quais situações surge a necessidade de regularização?

### Resposta de sabatina

A necessidade de regularização pode surgir quando o modelo apresenta overfitting, quando temos muitas variáveis, multicolinearidade ou coeficientes muito instáveis. A regularização penaliza a magnitude dos coeficientes e reduz a complexidade efetiva do modelo, buscando melhorar sua generalização. Dependendo do objetivo, também podemos utilizar Lasso quando queremos seleção de features.

### Exemplo prático

Resultados hipotéticos, medidos por RMSE na mesma unidade do alvo:

| Modelo | Treino | Validação |
|---|---:|---:|
| Sem regularização | 100 | 450 |
| Com regularização ajustada | 180 | 300 |

O erro de treino aumentou, mas o de validação caiu. Esse é o benefício procurado ao aceitar mais viés para reduzir variância.

## Pergunta 18

Por que a regressão linear minimiza o quadrado dos erros, em vez de simplesmente minimizar os erros?

### Resposta de sabatina

Porque, se simplesmente somássemos os resíduos, erros positivos e negativos poderiam se cancelar e dar a impressão de que o erro total é pequeno ou até zero. Ao elevar os resíduos ao quadrado, eliminamos esse cancelamento e também penalizamos mais fortemente os erros maiores. Além disso, a soma dos quadrados gera uma função de otimização conveniente para estimar os coeficientes da regressão. Na regressão linear, essa função é convexa e diferenciável. Sob erros gaussianos independentes com variância constante, minimizar a soma dos quadrados equivale a estimar os coeficientes por máxima verossimilhança. Também existem regressões com outras perdas, como a perda absoluta.

### Exemplo prático

Para resíduos `+10` e `−10`:

- Soma simples: `10 − 10 = 0`, apesar de duas previsões incorretas.
- Soma dos quadrados: `10² + (−10)² = 200`.
- Soma dos valores absolutos: `|10| + |−10| = 20`.

As duas últimas evitam cancelamento, mas tratam erros grandes de forma diferente: dobrar um erro de 10 para 20 quadruplica sua contribuição quadrática, de 100 para 400.

## Pergunta 19

Quais são as principais métricas utilizadas para avaliar uma regressão?

### Resposta de sabatina

As principais são MAE, MSE, RMSE, MAPE e R². O MAE é a média dos erros absolutos; o MSE, a média dos erros ao quadrado; e o RMSE, a raiz do MSE, na unidade do alvo. MSE e RMSE dão mais peso aos erros grandes. O MAPE mede o erro absoluto percentual, mas não é adequado com alvos iguais ou muito próximos de zero.

O R² compara a soma dos quadrados dos resíduos com a variação do alvo em torno da sua média: `R² = 1 − SSE/SST`. Pode ser negativo quando o modelo é pior que essa referência; com alvo constante, a fórmula fica indefinida. O R² ajustado penaliza o acréscimo de preditores, mas não substitui a avaliação fora da amostra.

### Exemplo prático

Valores reais `[100, 200]` e previstos `[90, 230]`:

| Métrica | Cálculo | Resultado |
|---|---|---:|
| MAE | `(10 + 30)/2` | 20 |
| MSE | `(10² + 30²)/2` | 500 |
| RMSE | `√500` | ≈ 22,36 |
| MAPE | `(10/100 + 30/200)/2 × 100` | 12,5% |
| R² | `1 − 1.000/5.000` | 0,80 |

No R², a média real é 150 e `SST = (100 − 150)² + (200 − 150)² = 5.000`.

## Pergunta 20

Em um modelo para prever a renda de clientes, com rendas muito diferentes entre si, qual métrica você utilizaria?

### Resposta de sabatina

Eu poderia utilizar MAE se o objetivo for interpretar o erro diretamente em reais, porque ele é simples de interpretar e menos sensível a erros extremos do que MSE ou RMSE. Porém, como as rendas são muito diferentes entre os clientes, se o negócio estiver interessado no erro relativo à renda, eu também avaliaria uma métrica percentual como MAPE, desde que não existam rendas iguais ou muito próximas de zero.

### Exemplo prático

O mesmo erro pode ter pesos relativos distintos:

| Cliente | Renda real | Previsão | Erro absoluto | Erro percentual |
|---|---:|---:|---:|---:|
| A | R$ 2.000 | R$ 3.000 | R$ 1.000 | 50% |
| B | R$ 100.000 | R$ 101.000 | R$ 1.000 | 1% |

O MAE desses dois casos é R$ 1.000; o MAPE é 25,5%. A escolha depende de avaliar o impacto em reais ou proporcionalmente à renda.

## Pergunta 21

O que você quis dizer ao comparar a renda de um mês com a renda de outro mês?

### Resposta de sabatina

A comparação distingue variação absoluta de variação relativa. Passar de R$ 2.000 para R$ 3.000 representa R$ 1.000 ou 50%; passar de R$ 100.000 para R$ 101.000 representa os mesmos R$ 1.000, mas apenas 1%.

Na avaliação do modelo, aplico essa mesma lógica à diferença entre renda real e prevista: MAE expressa o erro em reais; MAPE, proporcionalmente à renda real. A variação entre meses, isoladamente, não é um erro de previsão.

### Exemplo prático

Separe mudança no tempo de erro de previsão:

- Janeiro: renda real de R$ 2.000.
- Fevereiro: renda real de R$ 3.000 → aumento mensal de R$ 1.000, ou 50%.
- Previsão para fevereiro: R$ 2.700 → erro absoluto de R$ 300, ou 10% da renda real de fevereiro.

O aumento de 50% não é o erro do modelo; a previsão deve ser comparada ao valor real do período previsto.

## Pergunta 22

Como funciona uma árvore de decisão?

### Resposta de sabatina

Uma árvore de decisão divide os dados recursivamente por regras sobre as variáveis. Em cada nó, procura de forma gulosa a divisão que mais reduz a impureza, como Gini ou entropia em classificação, ou uma perda, como o erro quadrático em regressão. O processo termina em folhas, conforme os critérios de parada ou poda. Na folha, prevê normalmente a classe majoritária em classificação ou a média do alvo em regressão com perda quadrática. Assim, captura relações não lineares e interações.

### Exemplo prático

Uma árvore ilustrativa para inadimplência poderia produzir as regras:

```text
Possui atraso recente?
├── Sim → folha: 80% inadimplentes → prevê inadimplência
└── Não → renda ≤ R$ 3.000?
    ├── Sim → folha: 40% inadimplentes → prevê adimplência
    └── Não → folha: 10% inadimplentes → prevê adimplência
```
As classes acima usam limiar de 50%, sem pesos. As divisões e proporções seriam aprendidas com os dados de treino.

## Pergunta 23

Quais cuidados devem ser considerados ao utilizar uma árvore de decisão?

### Resposta de sabatina

O principal cuidado é controlar a complexidade da árvore, porque uma árvore muito profunda pode aprender ruídos da base de treinamento e sofrer overfitting, apresentando ótimo desempenho no treino e pior generalização no teste. Podemos controlar isso com hiperparâmetros como profundidade máxima, número mínimo de amostras para realizar um split, número mínimo de amostras por folha e também técnicas de poda. Além disso, árvores individuais apresentam alta variância e podem mudar bastante com pequenas alterações nos dados.

### Exemplo prático

Se uma árvore obtiver 100% de acurácia no treino e 72% na validação:

1. Investigar possível overfitting e a qualidade da separação dos dados.
2. Testar menor profundidade, maior mínimo de amostras por folha ou poda.
3. Comparar o resultado na validação com a mesma métrica.

Esses números são ilustrativos; a diferença entre treino e validação é um sinal a investigar.

## Pergunta 24

Qual é a diferença entre o índice Gini e a entropia?

### Resposta de sabatina

Gini e entropia são critérios utilizados para medir a impureza dos nós e escolher os splits de uma árvore de classificação. O Gini utiliza as proporções das classes em um cálculo mais simples, enquanto a entropia vem da teoria da informação e utiliza logaritmos para medir a incerteza do nó. Em ambos, zero representa um nó puro. Na prática, os dois frequentemente produzem resultados semelhantes, embora possam escolher divisões diferentes. As fórmulas são `Gini = 1 − Σpₖ²` e `Entropia = −Σpₖ log₂(pₖ)`, com `0 log₂(0) = 0`. A divisão é avaliada pela redução da impureza, ponderando os nós filhos pelo seu tamanho.

### Exemplo prático

Em classificação binária:

| Composição do nó | Gini | Entropia em bits |
|---|---:|---:|
| 100% A, 0% B | 0 | 0 |
| 50% A, 50% B | 0,50 | 1 |
| 80% A, 20% B | 0,32 | ≈ 0,722 |

Se um nó 50/50 for dividido em dois nós puros, a impureza ponderada dos filhos será zero: a redução de Gini será 0,50, e o ganho de informação será 1 bit.

## Pergunta 25

O que pode acontecer se não definirmos a profundidade máxima da árvore?

### Resposta de sabatina

Se não controlarmos a profundidade máxima, a árvore pode crescer excessivamente e criar regras cada vez mais específicas para os dados de treinamento. Com isso, aumenta sua complexidade e sua variância, podendo aprender ruídos da amostra e sofrer overfitting, apresentando ótimo desempenho no treino e pior generalização em dados novos.

### Exemplo prático

Uma árvore sem limite de profundidade e com demais critérios permissivos pode criar uma folha com apenas um cliente.

- Nesse cliente, acerta perfeitamente no treino.
- Para um cliente novo parecido, a regra pode não generalizar.
- Limitar a profundidade ou podar reduz a possibilidade de regras tão específicas.

Não definir `max_depth` não implica crescimento infinito: outros critérios e os próprios dados também limitam a árvore.

## Pergunta 26

O que pode acontecer se não definirmos um número mínimo de amostras por nó ou por folha?

### Resposta de sabatina

Se não controlarmos o número mínimo de amostras, a árvore pode realizar divisões com pouquíssimas observações e criar folhas muito específicas. Com isso, ela pode começar a modelar particularidades e ruídos da amostra de treinamento, aumentando sua variância e o risco de overfitting.

### Exemplo prático

Considere um nó com 20 amostras e uma divisão proposta em folhas de 19 e 1:

- Com `min_samples_leaf = 1`, essa divisão pode ser aceita.
- Com `min_samples_leaf = 5`, ela é rejeitada, pois uma folha ficaria com apenas uma amostra.
- `min_samples_split = 10` sozinho não impede a folha unitária: ele verifica o tamanho do nó antes da divisão.

## Pergunta 27

Quando uma folha ainda contém observações de classes diferentes, como a árvore escolhe a classe prevista?

### Resposta de sabatina

Quando uma folha contém observações de classes diferentes, a árvore normalmente prevê a classe majoritária daquela folha. Além disso, a proporção das classes pode ser utilizada para estimar a probabilidade. Por exemplo, se uma folha possui 70% de inadimplentes, a previsão será inadimplente e a probabilidade estimada dessa classe será aproximadamente 70%. Se houver pesos de amostra ou de classe, essa contagem pode ser ponderada.

### Exemplo prático

Uma folha contém 10 clientes, sem pesos:

- 7 inadimplentes e 3 adimplentes.
- Probabilidade estimada de inadimplência: `7/10 = 70%`.
- Classe majoritária: **inadimplente**.

Um limiar operacional de 80% mudaria a decisão binária, mesmo com a mesma probabilidade estimada.

## Pergunta 28

O output de uma árvore de classificação é apenas um valor binário?

### Resposta de sabatina

Não. Uma árvore de classificação pode retornar tanto a classe prevista quanto as probabilidades associadas às classes. Em um problema binário, por exemplo, ela pode retornar uma probabilidade de 80% para a classe 1 e, a partir disso, classificá-la como classe 1. Além disso, árvores também podem trabalhar com problemas multiclasse.

### Exemplo prático

Uma árvore multiclasse pode retornar:

| Classe | Probabilidade estimada |
|---|---:|
| Baixo risco | 0,20 |
| Médio risco | 0,65 |
| Alto risco | 0,15 |

A classe prevista pela maior probabilidade é “médio risco”, mas a saída probabilística preserva as três estimativas.

## Pergunta 29

O que representa a probabilidade fornecida por uma árvore de decisão?

### Resposta de sabatina

Em uma árvore de classificação, a probabilidade normalmente é estimada pela proporção das classes das observações de treinamento que chegaram àquela folha. Por exemplo, se 80% das observações da folha são da classe 1, a árvore pode retornar aproximadamente 80% para essa classe. Depois, um threshold pode ser utilizado para transformar essa probabilidade em uma classificação. Com pesos, usamos proporções ponderadas. Essa estimativa pode ser pouco confiável em folhas pequenas e não é necessariamente bem calibrada.

### Exemplo prático

Duas folhas sem pesos podem retornar a mesma estimativa:

| Folha | Positivos | Total | Probabilidade |
|---|---:|---:|---:|
| A | 4 | 5 | 80% |
| B | 80 | 100 | 80% |

A proporção é igual, mas a folha A tem muito menos suporte amostral. A estimativa de 80% não garante que exatamente 80% dos novos casos serão positivos.

## Pergunta 30

Existe algum tratamento específico que deve ser feito nas variáveis antes de usar uma árvore?

### Resposta de sabatina

Árvores geralmente exigem menos pré-processamento. Como os splits são baseados em regras e thresholds, normalmente não precisamos normalizar ou padronizar as variáveis. Elas também são relativamente robustas a outliers, embora valores extremos devam ser investigados quando representam erros. Os principais cuidados são valores ausentes e variáveis categóricas, dependendo do suporte da implementação utilizada.

### Exemplo prático

Converter renda de reais para milhares preserva a divisão equivalente:

- Antes: `renda ≤ 5.000`.
- Depois: `renda_em_milhares ≤ 5`.

Os mesmos clientes seguem para cada lado. Já um campo de renda ausente exige tratamento compatível com a implementação; mudar a escala não resolve a ausência.

## Pergunta 31

Como uma árvore trabalha com uma variável categórica?

### Resposta de sabatina

Uma árvore pode utilizar uma variável categórica para criar grupos de categorias buscando a divisão que mais reduz a impureza. Porém, isso depende da implementação. Alguns algoritmos possuem suporte nativo a categóricas, enquanto outros exigem encoding, como one-hot encoding. Também devemos tomar cuidado para não atribuir números arbitrários a categorias nominais e introduzir uma ordem que não existe.

### Exemplo prático

Para “canal de aquisição” com categorias loja, aplicativo e parceiro:

- Uma implementação com suporte categórico pode avaliar `{loja, parceiro}` versus `{aplicativo}`.
- Com one-hot encoding, uma divisão pode testar `é_aplicativo ≤ 0,5`.
- Com códigos arbitrários `loja = 1`, `aplicativo = 2`, `parceiro = 3`, uma única divisão numérica não separa `{1, 3}` de `{2}`.

## Pergunta 32

Como você representaria no modelo uma variável binária como “empregado” e “desempregado”?

### Resposta de sabatina

Como existem apenas duas categorias, eu representaria a variável de forma binária, por exemplo empregado igual a 1 e desempregado igual a 0. Poderíamos inverter os valores sem problema, desde que isso seja documentado. Nesse caso, não estamos introduzindo uma hierarquia artificial entre várias categorias.

### Exemplo prático

Uma única coluna é suficiente:

| Situação | Empregado |
|---|---:|
| Empregado | 1 |
| Desempregado | 0 |

Uma árvore pode separar as categorias com `empregado ≤ 0,5`. “Não informado” precisa de tratamento próprio; não deve ser automaticamente confundido com desempregado.

## Pergunta 33

Quais são os principais problemas de uma árvore de decisão?

### Resposta de sabatina

Os principais problemas de uma árvore de decisão são a alta variância e o risco de overfitting. Pequenas alterações na amostra de treinamento podem gerar árvores bastante diferentes, tornando uma árvore individual instável. Além disso, se permitirmos que ela cresça excessivamente, pode aprender ruídos do treinamento e perder capacidade de generalização. Por isso é importante controlar sua complexidade.

### Exemplo prático

Imagine duas amostras muito semelhantes:

- Na primeira, a melhor divisão inicial é `renda ≤ 4.000`.
- Após substituir poucos registros, outra variável passa a ser escolhida na raiz.
- Todas as divisões seguintes podem mudar, produzindo previsões diferentes.

Essa sensibilidade à amostra ilustra a alta variância de uma árvore individual.

## Pergunta 34

Qual é a diferença entre uma árvore de decisão e um Random Forest?

### Resposta de sabatina

Uma árvore de decisão é um modelo individual. O Random Forest combina várias árvores, normalmente treinadas em amostras bootstrap, e sorteia um subconjunto de variáveis candidatas a cada divisão. Na regressão, agrega as previsões pela média; na classificação, pode usar votação ou a média das probabilidades, conforme a implementação. A diversidade das árvores reduz sua correlação e permite diminuir principalmente a variância da previsão.

### Exemplo prático

Em regressão, três árvores preveem R$ 2.000, R$ 2.300 e R$ 2.600 para o mesmo cliente.

- Uma árvore individual fornece apenas sua própria previsão.
- A floresta agrega: `(2.000 + 2.300 + 2.600)/3 = R$ 2.300`.
- As árvores variam pelas amostras bootstrap e pelas variáveis candidatas a cada divisão.

## Pergunta 35

O que o Random Forest resolve em relação a uma única árvore de decisão?

### Resposta de sabatina

Uma árvore individual possui alta variância e pode mudar bastante com pequenas alterações nos dados. O Random Forest reduz principalmente essa variância combinando várias árvores treinadas com amostras bootstrap e utilizando subconjuntos aleatórios de features nos splits. Isso reduz a correlação entre as árvores e, ao agregar suas previsões, produz um modelo mais estável e geralmente menos suscetível ao overfitting do que uma árvore individual.

### Exemplo prático

Para B previsões com variância igual σ² e correlação par a par ρ, a variância da média é `σ²[ρ + (1 − ρ)/B]`.

- Com `B = 100` e `ρ = 0`, ela vale `0,01σ²`.
- Com `B = 100` e `ρ = 0,8`, ela vale `0,802σ²`.

O exemplo simplificado mostra por que diversidade importa: muitas árvores quase idênticas reduzem pouco a variância.

## Pergunta 36

Quais hiperparâmetros são interessantes de ajustar em um Random Forest?

### Resposta de sabatina

Os principais hiperparâmetros que eu ajustaria são `n_estimators`, que controla a quantidade de árvores; `max_depth`, que controla a profundidade; `min_samples_split` e `min_samples_leaf`, que controlam o crescimento das árvores; `max_features`, que determina quantas features podem ser consideradas em cada split; e `bootstrap`, que controla a utilização da amostragem com reposição. Em classificação desbalanceada, também avaliaria `class_weight`.

### Exemplo prático

Uma busca inicial ilustrativa pode comparar:

| Parâmetro | Valores candidatos | Objetivo |
|---|---|---|
| `n_estimators` | 200, 500 | Estabilidade versus custo |
| `max_depth` | 5, 10, sem limite | Complexidade |
| `min_samples_leaf` | 1, 5, 20 | Evitar folhas muito pequenas |
| `max_features` | Raiz do total, metade do total | Diversidade entre árvores |

Os valores são candidatos, não configurações universalmente recomendadas; a seleção depende da validação e do orçamento computacional.

## Pergunta 37

Se treinarmos uma árvore tradicional mil vezes com a mesma base, a mesma amostra e os mesmos parâmetros, ela dará sempre o mesmo resultado?

### Resposta de sabatina

Se a árvore for treinada com exatamente os mesmos dados, mesmos hiperparâmetros e sem aleatoriedade, ela produzirá a mesma estrutura e o mesmo resultado todas as vezes. Se a implementação possuir componentes aleatórios, precisamos fixar o `random_state` para garantir reprodutibilidade. No Random Forest, criamos árvores diferentes principalmente por meio das amostras bootstrap e da seleção aleatória de features, sem precisar utilizar hiperparâmetros diferentes para cada árvore.

### Exemplo prático

No mesmo ambiente e com os mesmos dados:

- Repetir uma árvore determinística mil vezes reproduz o mesmo modelo.
- Se houver desempates aleatórios, fixar a semente permite reproduzir o procedimento.
- Copiar mil vezes a mesma árvore não cria diversidade: a média de mil previsões idênticas é a própria previsão original.

## Pergunta 38

Qual é o ponto negativo de substituir uma árvore de decisão por um Random Forest?

### Resposta de sabatina

O principal ponto negativo é a perda de interpretabilidade. Uma árvore individual pode ser visualizada e suas regras podem ser acompanhadas diretamente, enquanto um Random Forest combina a previsão de muitas árvores, tornando o comportamento global mais difícil de interpretar. Além disso, existe um aumento no custo computacional e no uso de memória.

### Exemplo prático

Para explicar uma decisão:

- Em uma árvore de profundidade 3, é possível acompanhar um caminho com até três regras.
- Em uma floresta de 500 árvores, a previsão combina centenas de caminhos.
- Além de dificultar a explicação direta, a floresta precisa armazenar e percorrer muito mais nós.

## Pergunta 39

Quais algoritmos de boosting você conhece?

### Resposta de sabatina

Conheço AdaBoost, Gradient Boosting, XGBoost, LightGBM e CatBoost. Todos trabalham com a ideia de boosting, combinando modelos sequencialmente, embora tenham diferenças na forma de corrigir erros, otimizar a função de perda e construir as árvores.

### Exemplo prático

Exemplos de mecanismos ou aplicações:

| Método | Exemplo de característica |
|---|---|
| AdaBoost | Reponderar exemplos mal classificados no caso clássico |
| Gradient Boosting | Ajustar novos modelos aos gradientes negativos da perda |
| XGBoost | Boosting de árvores com objetivo regularizado |
| LightGBM | Construção eficiente de árvores usando histogramas |
| CatBoost | Estratégias específicas para variáveis categóricas e ordered boosting |

São exemplos para distinguir os métodos, sem implicar que um seja sempre superior aos demais.

## Pergunta 40

Por que um algoritmo de boosting poderia apresentar um resultado melhor?

### Resposta de sabatina

O boosting pode apresentar um resultado melhor porque constrói modelos sequencialmente, e cada novo modelo é adicionado buscando reduzir os erros do ensemble atual. Com isso, vários modelos simples podem ser combinados para formar um modelo mais complexo e reduzir principalmente o viés. No Gradient Boosting, essa correção ocorre buscando reduzir a função de perda. Também precisamos controlar hiperparâmetros como learning rate, número de árvores e profundidade para evitar excesso de complexidade e overfitting.

### Exemplo prático

Imagine que um modelo simples subestime sistematicamente a renda de um grupo de clientes.

1. A previsão inicial captura o padrão geral.
2. Um novo modelo aprende uma correção positiva para esse grupo.
3. Outros modelos corrigem padrões ainda não explicados pelo conjunto.

O ganho esperado deve ser confirmado na validação; continuar adicionando modelos pode reduzir a perda de treino e piorar a generalização.

## Pergunta 41

Qual é o “pulo do gato” do boosting?

### Resposta de sabatina

O pulo do gato do boosting é o aprendizado sequencial. Diferentemente do bagging, em que as árvores podem ser treinadas independentemente, no boosting cada novo modelo é construído buscando melhorar os erros do ensemble atual. No Gradient Boosting, cada nova árvore é adicionada na direção que reduz a função de perda. O learning rate controla o quanto cada nova árvore contribui para essa atualização.

### Exemplo prático

Para um cliente, suponha previsão atual de 100 e correção aprendida de +20:

- Com `learning rate = 1`, a nova previsão seria `100 + 1 × 20 = 120`.
- Com `learning rate = 0,1`, seria `100 + 0,1 × 20 = 102`.

O learning rate controla o tamanho do passo. As próximas etapas trabalham sobre o ensemble já atualizado.

## Pergunta 42

O que o boosting faz de diferente durante o treinamento para melhorar os resultados?

### Resposta de sabatina

O boosting adiciona modelos sequencialmente para melhorar o ensemble atual. No AdaBoost clássico, aumenta a importância dos exemplos classificados incorretamente. No Gradient Boosting, ajusta o novo modelo aos pseudo-resíduos, isto é, ao gradiente negativo da perda em relação à previsão atual. Para perda quadrática, eles correspondem aos resíduos, a menos de um fator de escala.

A atualização é `Fₘ(x) = Fₘ₋₁(x) + ηhₘ(x)`, em que η é o learning rate e hₘ é a contribuição do novo modelo, incluindo eventual ajuste de seu passo. Assim, cada etapa busca reduzir a perda do conjunto acumulado.

### Exemplo prático

Para perda quadrática, considere alvos `[10, 20, 30]` e previsão inicial constante `[20, 20, 20]`:

1. Resíduos: `[-10, 0, 10]`.
2. Suponha que a nova árvore consiga prever exatamente esses resíduos.
3. Com `η = 0,1`, a previsão atualizada é `[19, 20, 21]`.
4. A soma dos erros quadráticos cai de `200` para `162`.

Em outras perdas, como a logística, o alvo de cada etapa é o pseudo-resíduo correspondente, não necessariamente `y − ŷ` na mesma forma desse exemplo.

## Pergunta 43

Quais são as principais métricas de classificação?

### Resposta de sabatina

As principais incluem acurácia, precisão (precision), sensibilidade (recall), especificidade e F1-score, calculadas a partir da matriz de confusão em um ponto de decisão. Para avaliar os scores ao longo de diferentes limiares, usamos AUC-ROC e métricas da curva precisão–recall, como PR-AUC ou Average Precision, que têm definições de integração distintas. A escolha depende do desbalanceamento e dos custos de falsos positivos e falsos negativos.

### Exemplo prático

Em 10.000 transações, existem 100 fraudes. Um modelo que prevê “legítima” para todas apresenta:

- Accuracy: `9.900/10.000 = 99%`.
- Recall de fraude: `0/100 = 0%`.
- Precision de fraude: indefinida, pois nenhuma transação foi prevista como fraude.

A acurácia elevada esconde a incapacidade de encontrar o evento de interesse.

## Pergunta 44

Como funcionam accuracy, precision, recall, F1-score, curva ROC e AUC?

### Resposta de sabatina

Considerando TP como verdadeiros positivos, TN como verdadeiros negativos, FP como falsos positivos e FN como falsos negativos:

- **Accuracy:** `(TP + TN)/(TP + TN + FP + FN)`, a proporção total de acertos.
- **Precision:** `TP/(TP + FP)`, a proporção de positivos reais entre os classificados como positivos.
- **Recall:** `TP/(TP + FN)`, a proporção dos positivos reais encontrados.
- **F1-score:** `2TP/(2TP + FP + FN)`, a média harmônica entre precision e recall.
- **Curva ROC:** relaciona recall, ou TPR, com `FPR = FP/(FP + TN)` ao variar o limiar de decisão.
- **AUC-ROC:** área sob essa curva; mede a capacidade de ordenar positivos acima de negativos. Vale 0,5 para um ranking aleatório em expectativa e 1 para discriminação perfeita.

As razões exigem denominadores não nulos; casos indefinidos precisam de uma convenção explícita. A AUC avalia discriminação, mas não garante probabilidades calibradas.

### Exemplo prático

Considere a matriz de confusão:

| Real / Previsto | Positivo | Negativo |
|---|---:|---:|
| Positivo | TP = 40 | FN = 10 |
| Negativo | FP = 20 | TN = 930 |

- Accuracy: `(40 + 930)/1.000 = 97%`.
- Precision: `40/60 ≈ 66,7%`.
- Recall: `40/50 = 80%`.
- F1: `80/(80 + 20 + 10) ≈ 72,7%`.
- FPR: `20/950 ≈ 2,1%`; este limiar gera o ponto ROC `(2,1%; 80%)`.

Uma única matriz não determina a AUC: precisamos dos scores para avaliar a ordenação ao longo dos limiares.

## Pergunta 45

Em um modelo de fraude, no qual o evento é raro e queremos capturar as fraudes sem bloquear todas as transações, quais métricas devem ser priorizadas?

### Resposta de sabatina

Nesse problema eu avaliaria principalmente Precision e Recall em conjunto. Recall é importante para capturar o maior número possível de fraudes e reduzir falsos negativos, enquanto Precision é importante para evitar muitos falsos positivos e o bloqueio de transações legítimas. Como existe um trade-off entre essas métricas, eu ajustaria o threshold de acordo com os custos de negócio e também poderia acompanhar métricas como PR-AUC.

### Exemplo prático

Para 100 fraudes reais, dois limiares produzem:

| Limiar | TP | FP | Recall | Precision |
|---|---:|---:|---:|---:|
| Mais alto | 60 | 10 | 60% | 85,7% |
| Mais baixo | 90 | 90 | 90% | 50% |

O limiar mais baixo captura 30 fraudes adicionais, mas bloqueia 80 transações legítimas a mais. A decisão depende dos valores envolvidos e do custo desses bloqueios.

## Pergunta 46

Se o modelo classificou somente uma transação como fraude e ela realmente era fraude, qual seria a precision?

### Resposta de sabatina

A Precision seria 100%, porque o modelo classificou uma transação como fraude e ela realmente era fraude. Portanto, tivemos um verdadeiro positivo e nenhum falso positivo.

### Exemplo prático

Cálculo do cenário:

- Transações sinalizadas: 1.
- Fraudes entre as sinalizadas: 1 → `TP = 1`, `FP = 0`.
- `Precision = 1/(1 + 0) = 100%`.

Esse resultado não informa quantas fraudes passaram despercebidas; para isso, precisamos do recall.

## Pergunta 47

Se existiam cem fraudes reais, mas o modelo encontrou apenas uma, o que isso representa em termos de recall?

### Resposta de sabatina

O Recall seria de 1%, porque existiam 100 fraudes reais e o modelo encontrou apenas uma. Portanto, ele capturou somente 1% dos eventos positivos existentes.

### Exemplo prático

Se há 100 fraudes e só uma foi encontrada:

- `TP = 1` e `FN = 99`.
- `Recall = 1/(1 + 99) = 1%`.
- O modelo deixou passar 99% das fraudes.

Combinando este cenário ao da pergunta anterior, precision pode ser 100% e recall apenas 1% ao mesmo tempo.

## Pergunta 48

Em uma base de dois milhões de clientes, se você puder selecionar apenas os cem clientes com maior probabilidade de contratar um seguro de vida, qual métrica utilizaria para validar o modelo?

### Resposta de sabatina

Como só posso selecionar 100 clientes entre 2 milhões, eu avaliaria o modelo principalmente com Precision@100. Ela mede, entre os 100 clientes com maior score ou probabilidade prevista pelo modelo, quantos realmente contrataram o produto. Também poderia acompanhar Lift@100 para verificar o ganho em relação a uma seleção aleatória.

### Exemplo prático

Em uma base de avaliação, 20 dos 100 clientes com maiores scores são compradores:

- `Precision@100 = 20/100 = 20%`.
- Se a taxa de compradores na base é 2%, uma seleção aleatória de 100 teria, em média, 2 compradores.
- `Lift@100 = 20%/2% = 10`.

Isso representa concentração de compradores dez vezes maior no topo. A métrica de ranking, sozinha, não mede o efeito causal de entrar em contato com o cliente.

## Pergunta 49

Por que não utilizar somente AUC nesse problema?

### Resposta de sabatina

Eu não utilizaria somente AUC porque ela avalia a capacidade de discriminação do modelo de forma global e ao longo de diferentes thresholds. Nesse problema, o negócio só vai atuar sobre os 100 clientes com maior score. Portanto, preciso de uma métrica focada no topo do ranking, como Precision@100, para saber quantos desses 100 clientes realmente são positivos.

### Exemplo prático

Resultados hipotéticos em uma mesma base de avaliação:

| Modelo | AUC-ROC | Compradores no top 100 | Precision@100 |
|---|---:|---:|---:|
| A | 0,92 | 15 | 15% |
| B | 0,88 | 25 | 25% |

Se a ação é limitada a cem contatos e cada conversão tem o mesmo valor, B entrega o melhor topo do ranking, apesar da AUC global menor.

## Pergunta 50

Por que não utilizar somente F1-score nesse problema?

### Resposta de sabatina

Eu não utilizaria somente F1 porque ele busca equilibrar Precision e Recall, enquanto nesse problema existe uma restrição operacional de selecionar apenas 100 clientes. Mesmo que os 100 escolhidos fossem todos compradores, o Recall poderia continuar muito baixo se existissem milhares de compradores na base. Por isso, uma métrica de ranking como Precision@100 representa melhor o objetivo do negócio.

### Exemplo prático

Suponha 10.000 compradores reais na base e um top 100 contendo somente compradores:

- Precision@100: `100/100 = 100%`.
- Recall: `100/10.000 = 1%`.
- F1 ao classificar só esses cem como positivos: `2 × 1 × 0,01/(1 + 0,01) ≈ 1,98%`.

O F1 é baixo, embora o resultado seja perfeito dentro da capacidade de cem contatos. Com K e o total de positivos fixos, F1 e Precision@K ordenam modelos pelo mesmo número de acertos; o problema é usar F1 isoladamente sem explicitar essa restrição.

## Pergunta 51

Como funciona o algoritmo KNN?

### Resposta de sabatina

O KNN é um algoritmo baseado em proximidade. Quando chega uma nova observação, ele calcula sua distância em relação às observações de treino, seleciona os K vizinhos mais próximos e utiliza essas observações para fazer a previsão. Em classificação, normalmente utiliza voto majoritário; em regressão, a média dos vizinhos. Como é baseado em distância, é importante escalar as variáveis. Além disso, K pequeno tende a aumentar a variância e o risco de overfitting, enquanto K muito grande aumenta o viés e pode gerar underfitting. Ele também é considerado lazy learning, porque praticamente não há treinamento e boa parte do custo fica na predição.

### Exemplo prático

Com `K = 3`, os vizinhos mais próximos de um cliente têm:

- Classes `[inadimplente, adimplente, inadimplente]` → voto majoritário: **inadimplente**.
- Em uma tarefa de regressão, rendas `[2.000, 2.500, 3.000]` → previsão por média: **R$ 2.500**.

Com ponderação por distância, vizinhos mais próximos podem receber maior influência.

## Pergunta 52

Quais cuidados devem ser considerados ao utilizar o KNN?

### Resposta de sabatina

Como o KNN é baseado em distância, um dos principais cuidados é colocar as variáveis em escalas comparáveis. Também precisamos escolher adequadamente o K, porque valores pequenos aumentam a variância e valores muito grandes aumentam o viés. Outro problema é a alta dimensionalidade, que pode tornar as distâncias menos informativas. Também devemos tratar adequadamente variáveis categóricas e outliers e considerar o custo de predição, porque o KNN é um lazy learner e precisa consultar os dados de treino para classificar novas observações.

### Exemplo prático

Sem escala adequada, compare diferenças entre dois clientes:

- Idade: 10 anos → contribuição quadrática de `10² = 100`.
- Renda: R$ 10.000 → contribuição quadrática de `10.000² = 100.000.000`.

Na distância euclidiana bruta, a renda domina. Padronizar usando parâmetros aprendidos no treino reduz o efeito das unidades, embora a relevância das variáveis ainda precise ser avaliada.

## Pergunta 53

Como você escolheria o número de vizinhos (K) no KNN?

### Resposta de sabatina

Eu testaria diferentes valores de K utilizando validação cruzada e escolheria aquele que apresenta o melhor desempenho de validação segundo a métrica adequada ao problema. Também consideraria o trade-off entre viés e variância: K muito pequeno tende ao overfitting e K muito grande ao underfitting. Em classificação binária, também é comum testar valores ímpares para reduzir a possibilidade de empate.

### Exemplo prático

Resultados hipotéticos de validação cruzada:

| K | F1 médio |
|---|---:|
| 1 | 0,72 |
| 5 | 0,80 |
| 15 | 0,77 |

Entre esses candidatos, K = 5 apresenta o melhor resultado médio. Também avaliaria a variação entre as divisões e manteria o teste final separado da escolha.

## Pergunta 54

Qual heurística ou método utilizaria para escolher esse (K)?

### Resposta de sabatina

Eu utilizaria validação cruzada testando diferentes valores de K e escolheria aquele com melhor desempenho médio na métrica de interesse. Como heurística inicial, também podemos usar K próximo da raiz quadrada do número de observações, mas o valor final deve ser validado nos dados.

### Exemplo prático

Com `n = 10.000`, a heurística `√n` sugere K perto de 100.

1. Usar esse valor apenas como referência inicial.
2. Testar uma faixa, por exemplo `5, 15, 31, 61, 101, 151`.
3. Escolher pela validação, ajustando também a métrica de distância e a ponderação quando necessário.

A raiz quadrada não é uma regra de optimalidade; o melhor K pode ficar longe de 100.

## Pergunta 55

Como funciona o K-means?

### Resposta de sabatina

O K-means é um algoritmo não supervisionado de clusterização que divide as observações em K grupos. Inicialmente são definidos K centroides. Cada observação é associada ao centroide mais próximo e, depois, cada centroide é recalculado pela média dos pontos pertencentes ao seu cluster. Esse processo de atribuição e atualização é repetido até a convergência. O objetivo é minimizar as distâncias quadráticas dos pontos aos centroides dos seus respectivos clusters.

### Exemplo prático

Em uma dimensão, considere os pontos `[1, 2, 8, 9]`, `K = 2` e centroides iniciais 1 e 8.

1. Atribuição: `{1, 2}` ao primeiro centroide; `{8, 9}` ao segundo.
2. Atualização: médias `1,5` e `8,5`.
3. Nova atribuição: os grupos permanecem iguais → convergência.
4. Inércia final: `0,5² + 0,5² + 0,5² + 0,5² = 1`.

## Pergunta 56

Como escolher o número de clusters no K-means?

### Resposta de sabatina

Uma forma de escolher o número de clusters é o método do cotovelo. Eu treino o K-means para diferentes valores de K e analiso a soma dos quadrados das distâncias dos pontos aos seus respectivos centroides, chamada WCSS ou inércia. Conforme aumento K, essa medida diminui, e procuro o ponto em que adicionar novos clusters passa a gerar pouco ganho. Também posso complementar essa análise com o coeficiente de silhueta e com a interpretação de negócio.

### Exemplo prático

Resultados ilustrativos:

| K | Inércia |
|---|---:|
| 1 | 1.000 |
| 2 | 500 |
| 3 | 220 |
| 4 | 200 |
| 5 | 185 |

K = 3 é um candidato a “cotovelo”: depois dele, o ganho cai bastante. Eu confirmaria com silhueta, estabilidade e utilidade dos grupos. Nem todo conjunto possui um cotovelo claro.

## Pergunta 57

Como funciona o coeficiente de silhueta?

### Resposta de sabatina

A silhueta compara coesão e separação. Para cada ponto, `a` é a distância média aos demais pontos do próprio cluster; `b` é a menor distância média aos pontos de qualquer outro cluster. Calculamos `s = (b − a)/max(a, b)` e podemos tirar a média dos valores para avaliar o agrupamento.

Valores próximos de 1 indicam boa separação; próximos de zero, proximidade da fronteira; negativos sugerem que o ponto está, em média, mais próximo de outro grupo. Para clusters com um único ponto, usa-se convencionalmente silhueta zero. O cálculo exige pelo menos dois clusters e depende da métrica de distância.

### Exemplo prático

Para um ponto com `a = 2` e `b = 5`:

- `s = (5 − 2)/5 = 0,6` → boa separação relativa.

Para outro com `a = 5` e `b = 2`:
- `s = (2 − 5)/5 = −0,6` → ele está, em média, mais próximo de outro cluster.

Aqui, b usa a distância média a todos os pontos do cluster alternativo mais próximo, não apenas ao seu centroide.

## Pergunta 58

Em qual situação prática você utilizaria K-means?

### Resposta de sabatina

Eu poderia utilizar K-means para segmentação de clientes quando não tenho previamente os grupos rotulados. Por exemplo, utilizando renda, movimentação, quantidade de produtos e comportamento transacional, posso agrupar clientes semelhantes e depois analisar as características de cada cluster para criar estratégias específicas de negócio.

### Exemplo prático

Após padronizar frequência de compras e ticket médio, três grupos poderiam ser interpretados assim:

| Grupo | Frequência | Ticket médio | Possível interpretação |
|---|---|---|---|
| A | Alta | Baixo | Compras recorrentes de menor valor |
| B | Baixa | Alto | Compras ocasionais de maior valor |
| C | Alta | Alto | Clientes de alto valor observado |

Esses nomes são atribuídos após analisar os grupos; o K-means não conhece previamente esses perfis nem garante que eles aparecerão.

## Pergunta 59

Quais são os pontos negativos e as limitações do K-means?

### Resposta de sabatina

As principais limitações do K-means são a necessidade de definir K previamente, a sensibilidade à escala e aos outliers, a dependência da inicialização dos centroides e a dificuldade com dados categóricos. Além disso, ele funciona melhor com clusters compactos e bem separados, podendo ter dificuldade com formatos complexos, densidades diferentes e alta dimensionalidade. Como pode convergir para mínimos locais, também é importante utilizar boas estratégias de inicialização, como K-means++.

### Exemplo prático

Observe a sensibilidade da média a um outlier:

- Para o grupo `[1, 2, 3]`, o centroide é 2.
- Acrescentando 100, o centroide passa a `(1 + 2 + 3 + 100)/4 = 26,5`.

Um único ponto deslocou bastante o centroide. Com múltiplos clusters, também pode alterar as atribuições ou consumir um centroide para representar poucos casos extremos.

## Pergunta 60

O K-means é computacionalmente viável para uma base com aproximadamente quarenta milhões de clientes?

### Resposta de sabatina

Sim, pode ser viável, mas depende do número de observações, clusters, features, iterações e da infraestrutura disponível. Para uma base com 40 milhões de clientes, eu consideraria Mini-Batch K-means, que atualiza os centroides utilizando lotes menores e reduz o custo de processamento e memória. Também avaliaria redução de dimensionalidade e processamento distribuído, se necessário.

### Exemplo prático

Com 40 milhões de linhas e 20 variáveis em números de 32 bits:

- Matriz numérica bruta: `40.000.000 × 20 × 4 bytes = 3,2 GB` decimais.
- O uso real pode ser maior por cópias, estruturas auxiliares e pré-processamento.
- Mini-Batch pode atualizar centroides usando lotes menores; a redução de memória total depende de como a leitura e o armazenamento são implementados.

Eu faria um benchmark com amostra representativa e estimaria tempo, memória e qualidade antes de executar a carga completa.

## Pergunta 61

O treinamento do K-means pode ser paralelizado?

### Resposta de sabatina

O K-means pode ser bastante paralelizado, principalmente no cálculo das distâncias e na atribuição dos pontos aos clusters, porque diferentes partições dos dados podem ser processadas simultaneamente. Depois, os resultados parciais são agregados para recalcular os centroides. Porém, ele não é totalmente paralelizável, porque uma iteração depende dos centroides calculados na iteração anterior.

### Exemplo prático

Duas máquinas processam pontos atribuídos ao mesmo cluster:

| Máquina | Soma dos valores | Quantidade |
|---|---:|---:|
| A | 10 | 2 |
| B | 30 | 3 |

O centroide global é `(10 + 30)/(2 + 3) = 8`. A média simples dos centroides locais, `(5 + 10)/2 = 7,5`, estaria errada porque os grupos têm tamanhos diferentes. Depois da agregação, começa a próxima iteração.

## Pergunta 62

Qual é o custo computacional do treinamento do K-means?

### Resposta de sabatina

Para o algoritmo tradicional de Lloyd, o custo de treinamento é aproximadamente `O(n × K × d × i)`, em que n é o número de observações, K o número de clusters, d o número de variáveis e i o número de iterações. Em cada iteração, comparamos cada ponto com os K centroides usando suas d dimensões e atualizamos as médias. Se executarmos várias inicializações, o custo total também cresce com o número de execuções.

### Exemplo prático

Para `n = 1.000.000`, `K = 10`, `d = 20` e `i = 30`:

- Fator de trabalho: `n × K × d × i = 6.000.000.000`.
- Dobrar K de 10 para 20 aproximadamente dobra esse fator.
- Executar cinco inicializações aproximadamente multiplica o custo por cinco, mantendo os demais fatores.

Esse produto indica a ordem de trabalho; não é uma contagem exata de instruções nem uma estimativa direta de segundos.

## Pergunta 63

Depois que o K-means está treinado, escorar ou classificar um cliente novo é custoso ou rápido?

### Resposta de sabatina

A atribuição de um novo cliente é relativamente rápida: aplicamos o mesmo pré-processamento do treino, calculamos a distância aos K centroides e escolhemos o mais próximo. O custo dessa comparação é `O(K × d)` por observação, sem percorrer toda a base histórica nem recalcular os centroides. Trata-se de atribuir um cluster, e não de prever uma classe supervisionada.

### Exemplo prático

Para um ponto novo `x = (2, 3)` e centroides `C₁ = (1, 1)` e `C₂ = (5, 5)`:

- Distância quadrática a C₁: `(2 − 1)² + (3 − 1)² = 5`.
- Distância quadrática a C₂: `(2 − 5)² + (3 − 5)² = 13`.
- Atribuição: cluster 1.

Não é preciso calcular a raiz quadrada para comparar essas distâncias, nem consultar os pontos históricos.

## Pergunta 64

Em quais situações o K-means falha?

### Resposta de sabatina

O K-means pode falhar quando os clusters possuem formatos não convexos ou muito irregulares, densidades e tamanhos muito diferentes, ou quando existem muitos outliers. Ele também é sensível à escala, à inicialização dos centroides e pode sofrer em alta dimensionalidade. Isso ocorre porque ele funciona melhor com clusters compactos e razoavelmente bem separados.

### Exemplo prático

Imagine dois grupos em formato de meias-luas entrelaçadas:

- Cada grupo acompanha uma curva.
- O K-means atribui pontos pela proximidade a centroides, criando regiões de Voronoi convexas.
- Com K = 2, essa geometria pode cortar as luas, misturando partes dos grupos desejados.

Esse problema estrutural não é resolvido apenas aumentando o número de inicializações.

## Pergunta 65

Qual outro método de agrupamento poderia ser utilizado como alternativa ao K-means?

### Resposta de sabatina

Uma alternativa é o DBSCAN. Diferentemente do K-means, ele cria clusters com base na densidade dos pontos e não exige definir previamente o número de clusters. Além disso, consegue encontrar grupos com formatos mais irregulares e identificar determinadas observações como ruído ou outliers.

### Exemplo prático

Para coordenadas de pontos em dois anéis separados, com alguns pontos isolados:

- K-means pode dividir o espaço de forma incompatível com os anéis.
- DBSCAN pode seguir a continuidade das regiões densas e marcar isolados como ruído.
- Isso depende de `eps` e `min_samples` adequados; densidades muito diferentes podem dificultar o uso de um único eps.

## Pergunta 66

Como funciona o DBSCAN?

### Resposta de sabatina

O DBSCAN agrupa pontos por densidade usando `eps`, o raio da vizinhança, e `min_samples`, a quantidade mínima de pontos para definir um núcleo, contando o próprio ponto na convenção usual. Um ponto núcleo satisfaz esse mínimo; um ponto de borda não é núcleo, mas está na vizinhança de um núcleo; um ponto de ruído não satisfaz nenhuma dessas condições.

Os grupos se expandem por cadeias de pontos núcleo vizinhos e incluem suas bordas. O método não exige definir K e encontra formas irregulares, mas é sensível à escala, à métrica de distância e à escolha dos parâmetros.

### Exemplo prático

Em uma linha, use pontos `A = 0`, `B = 0,1`, `C = 0,2`, `D = 0,4` e `E = 2`, com `eps = 0,21` e `min_samples = 3`, contando o próprio ponto:

| Ponto | Pontos na vizinhança | Tipo |
|---|---|---|
| A | A, B, C | Núcleo |
| B | A, B, C | Núcleo |
| C | A, B, C, D | Núcleo |
| D | C, D | Borda |
| E | E | Ruído |

D entra no cluster por estar perto de C, embora não tenha pontos suficientes para ser núcleo.

## Pergunta 67

Como ocorre o processo de agrupamento ou treinamento do DBSCAN?

### Resposta de sabatina

O DBSCAN seleciona um ponto ainda não visitado e consulta sua vizinhança de raio `eps`. Se houver pelo menos `min_samples` pontos, incluindo o próprio ponto, ele é núcleo e inicia um cluster. O algoritmo adiciona seus vizinhos e continua a expansão a partir daqueles que também são núcleos, até esgotar a região densa. Pontos de borda entram no grupo, mas não propagam a expansão.

O processo se repete para os pontos restantes. Um ponto inicialmente marcado como ruído pode depois entrar como borda de um cluster; ao final, os pontos que não pertencem a nenhum grupo permanecem como ruído.

### Exemplo prático

Usando os pontos e parâmetros da pergunta 66:

1. Visitar A: sua vizinhança tem três pontos → iniciar um cluster.
2. Adicionar B e C; ambos também são núcleos.
3. Expandir a partir de C e incluir D.
4. D é borda, então não continua a expansão.
5. E fica isolado e termina como ruído.

Resultado: **cluster `{A, B, C, D}`; ruído `{E}`**. Se D fosse visitado primeiro, poderia ser marcado provisoriamente como ruído e depois incorporado como borda ao alcançar C.
