# Árvores de Decisão — Classificação e Regressão

> Resumo enxuto para sabatina: estrutura, critérios, regularização, custo e diferenças entre classificação e regressão.

## Visão rápida

| Aspecto | Árvore de decisão |
|---|---|
| Aprendizado | Supervisionado |
| Tarefas | Classificação e regressão |
| Natureza | Preditiva, não paramétrica e gulosa |
| Representação | Regras condicionais em um grafo acíclico |
| Escala | Não exige padronização |
| Comportamento típico | Baixo viés e alta variância quando cresce sem controle |

## Estrutura e funcionamento

- Nó raiz: primeiro corte.
- Nó interno: teste condicional em uma variável.
- Ramo: resultado do teste.
- Folha: previsão final.
- Caminho raiz–folha: conjunto de regras `se... então...`.

O algoritmo segue “dividir para conquistar”. Em cada nó, avalia candidatos de variável e limiar, escolhe localmente o corte que mais reduz a perda e repete o processo. Essa escolha local caracteriza o comportamento guloso: ela não garante a árvore globalmente ótima.

## Classificação x regressão

| Elemento | Classificação | Regressão |
|---|---|---|
| Target | Classe | Valor contínuo |
| Folha | Classe majoritária ou probabilidades | Valor numérico representativo |
| Critério comum | Gini ou entropia | Erro quadrático ou absoluto |
| Predição | Classe/probabilidade | Número |

Em regressão com erro quadrático, a folha normalmente prevê a média dos alvos que chegaram nela. Com erro absoluto, a mediana é a escolha natural.

## Como a árvore de regressão escolhe cortes

Para cada corte candidato, calcula-se a perda ponderada dos nós filhos:

```text
Perda do corte = (nL/n) · PerdaL + (nR/n) · PerdaR
```

No critério quadrático, cada nó tenta minimizar:

```text
SSE = Σ(yᵢ - ȳ_folha)²
```

O melhor corte é aquele com a maior redução de perda em relação ao nó pai.

No `DecisionTreeRegressor`, os critérios relevantes incluem:

- `squared_error`: redução de erro quadrático;
- `friedman_mse`: variação do erro quadrático usada em certos contextos;
- `absolute_error`: erro absoluto, mais robusto, porém mais custoso;
- `poisson`: adequado a alvos de contagem não negativos.

`gini` e `entropy` são critérios de classificação, não de regressão. Materiais antigos podem chamar `squared_error` de `mse` e `absolute_error` de `mae`.

## Profundidade e viés–variância

`max_depth` é o número máximo de níveis/cortes no caminho mais longo da raiz até uma folha.

| Árvore | Viés | Variância | Risco |
|---|---:|---:|---|
| Rasa/restrita | maior | menor | underfitting |
| Profunda/flexível | menor | maior | overfitting |

Uma árvore muito profunda cria muitas regras, folhas com poucos registros, erro de treino baixo e maior diferença entre treino e teste. Pequenas mudanças na amostra podem gerar uma estrutura bastante diferente.

## Como controlar overfitting

### Pré-poda

- `max_depth`;
- `max_leaf_nodes`;
- `min_samples_split`;
- `min_samples_leaf`;
- `min_impurity_decrease`;
- `max_features`.

### Pós-poda

- `ccp_alpha`: poda por complexidade de custo.

### Ensembles

- `RandomForestRegressor` e Extra Trees reduzem variância;
- Gradient Boosting e AdaBoost combinam árvores sequencialmente.

Os hiperparâmetros devem ser escolhidos por validação cruzada, não pelo desempenho no teste final.

## Dados, escala e outliers

Árvores usam ordenação e limiares, portanto transformações monotônicas de escala normalmente preservam os cortes. `StandardScaler` ou `MinMaxScaler` não são necessários para a árvore isolada.

Na teoria, árvores podem criar cortes em variáveis numéricas e categóricas. No scikit-learn, `DecisionTreeRegressor` e `DecisionTreeClassifier` exigem entrada numérica; categorias precisam ser codificadas. Uma codificação ordinal pode introduzir ordem artificial, então o tratamento depende do significado da variável.

Árvores são relativamente robustas a valores extremos nas features, mas não são imunes a outliers. Em regressão, outliers no target podem alterar fortemente médias de folhas e cortes com `squared_error`. Alternativas:

- investigar a natureza dos pontos;
- testar `absolute_error`;
- transformar o target quando fizer sentido;
- aumentar `min_samples_leaf`;
- usar métodos robustos ou ensembles.

Portanto, responder simplesmente “outliers não impactam árvores” é uma simplificação perigosa.

## Custo computacional

| Etapa | Custo típico |
|---|---|
| Treino | aproximadamente `O(p · n log n)` em condições usuais |
| Predição em árvore balanceada | `O(log n)` ou, de forma mais direta, `O(profundidade)` |
| Predição no pior caso | `O(n)` em árvore degenerada |
| Memória | proporcional ao número de nós |

Quando uma questão chama o custo de árvore de “logarítmico”, normalmente está falando da predição média em uma árvore balanceada. O treinamento não é simplesmente logarítmico.

## Parâmetros aprendidos x hiperparâmetros

### Aprendidos no treino

- variável escolhida em cada nó;
- limiar de cada corte;
- ligações entre nós;
- valor ou distribuição armazenada nas folhas.

### Definidos antes do treino

- `criterion`;
- `splitter`;
- `max_depth`;
- `min_samples_split` e `min_samples_leaf`;
- `max_features` e `max_leaf_nodes`;
- `min_impurity_decrease`;
- `ccp_alpha`;
- `random_state`.

## Pontos positivos

- regras fáceis de explicar e visualizar;
- modela relações não lineares e interações;
- pouco pré-processamento;
- não exige padronização;
- atende classificação e regressão;
- predição rápida.

## Pontos negativos e mitigação

| Limitação | Como mitigar |
|---|---|
| Alta variância/instabilidade | poda, validação cruzada e ensembles |
| Overfitting | limitar profundidade, folhas e amostras mínimas |
| Predições em degraus | ensembles ou modelo mais suave |
| Extrapolação ruim | não usar fora do domínio de treino sem cautela |
| Viés para certas representações | tratamento adequado de categorias e validação |

## Pegadinhas de sabatina

- Árvore de regressão continua sendo supervisionada, preditiva, não paramétrica e gulosa.
- A estrutura é a mesma da árvore de classificação; mudam a perda e a previsão na folha.
- `Gini` e entropia não são critérios de regressão.
- A saída de uma folha de regressão costuma ser a média, não uma classe.
- Não é preciso escalar os dados para uma árvore isolada.
- Profundidade grande não garante melhor teste: tende a reduzir viés e aumentar variância.
- “Custo logarítmico” descreve melhor a predição de uma árvore balanceada, não todo o treinamento.
- Uma árvore não extrapola: fora da região conhecida, continua retornando valores de folhas aprendidas.

## Resposta oral em 30 segundos

> Uma árvore de regressão é um modelo supervisionado e não paramétrico que particiona o espaço por regras condicionais. Em cada nó, escolhe de forma gulosa o corte que mais reduz uma perda de regressão, como erro quadrático; na folha, retorna um valor numérico, normalmente a média dos alvos. Árvores não exigem escala e capturam não linearidades, mas árvores profundas têm baixo viés, alta variância e podem sobreajustar. Controlamos isso com profundidade, amostras mínimas, poda e ensembles.
