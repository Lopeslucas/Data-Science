# Regressão Logística — resumo para sabatina

Este material reorganiza os conceitos cobrados em um guia contínuo. O objetivo é reunir o que costuma ser exigido em uma sabatina: funcionamento, premissas, parâmetros, hiperparâmetros, custo, limitações e pegadinhas.

---

## 1. Ideia central

A regressão logística é um algoritmo **supervisionado, preditivo, paramétrico e usado para classificação**.

Apesar do nome “regressão”, sua saída principal é a **probabilidade de uma classe**. Em um problema binário, o modelo estima, por exemplo:

> Qual é a probabilidade de este cliente pertencer à classe positiva, dados seus atributos?

Depois, um ponto de corte transforma a probabilidade em classe:

- se `P(y = 1 | x) >= limiar`, prevê classe 1;
- caso contrário, prevê classe 0.

O limiar costuma começar em `0,5`, mas **não é obrigatório**. Ele deve refletir o custo de falsos positivos e falsos negativos.

### Resposta curta para a sabatina

> A regressão logística modela o logaritmo da chance de um evento como uma combinação linear dos atributos. Em seguida, usa a função sigmoide para converter esse valor em uma probabilidade entre zero e um.

---

## 2. Como o modelo funciona

O modelo começa calculando um escore linear:

```text
z = beta_0 + beta_1*x_1 + beta_2*x_2 + ... + beta_p*x_p
```

Em que:

- `beta_0` é o intercepto;
- `beta_j` é o coeficiente associado ao atributo `x_j`;
- `z` pode assumir qualquer valor real.

Como `z` não é uma probabilidade, ele passa pela função sigmoide:

```text
P(y = 1 | x) = 1 / (1 + exp(-z))
```

Com isso:

- `z = 0` produz probabilidade `0,5`;
- `z` muito positivo produz probabilidade próxima de `1`;
- `z` muito negativo produz probabilidade próxima de `0`.

### Probabilidade, chance e logit

Esses três conceitos não são sinônimos:

```text
Probabilidade: p

Chance (odds): p / (1 - p)

Logit: log(p / (1 - p))
```

Exemplo: se `p = 0,80`:

```text
odds = 0,80 / 0,20 = 4
```

Isso significa uma chance de **4 para 1** a favor do evento, e não uma probabilidade de 400%.

A equação central do modelo é:

```text
log(p / (1 - p)) = beta_0 + beta_1*x_1 + ... + beta_p*x_p
```

Portanto, a regressão logística é **linear no logit e nos parâmetros**, não diretamente na probabilidade.

---

## 3. O que o modelo aprende

### Parâmetros aprendidos

São estimados durante o treinamento:

- intercepto `beta_0`;
- coeficientes `beta_1, ..., beta_p`.

Eles representam o conhecimento adquirido pelo modelo.

Mantendo as demais variáveis constantes, aumentar `x_j` em uma unidade altera o logaritmo da chance em `beta_j`. Em termos de chance:

```text
odds ratio = exp(beta_j)
```

- `exp(beta_j) > 1`: a chance do evento aumenta;
- `exp(beta_j) < 1`: a chance do evento diminui;
- `exp(beta_j) = 1`: não há alteração na chance.

### Hiperparâmetros

São escolhidos antes do ajuste e controlam o treinamento:

| Hiperparâmetro | Função prática |
|---|---|
| `penalty` | Define a regularização: L1, L2 ou Elastic Net |
| `C` | Inverso da força de regularização no scikit-learn |
| `solver` | Algoritmo numérico usado na otimização |
| `max_iter` | Limite de iterações do otimizador |
| `class_weight` | Aumenta ou reduz o peso das classes |
| `tol` | Tolerância usada no critério de convergência |
| `l1_ratio` | Mistura entre L1 e L2 no Elastic Net |

### Pegadinha sobre `C`

No scikit-learn:

```text
C pequeno  -> regularização mais forte
C grande   -> regularização mais fraca
```

É comum inverter essa interpretação.

---

## 4. Treinamento e regularização

Os coeficientes são normalmente estimados por **máxima verossimilhança**. Na prática, um otimizador numérico minimiza a log loss, também chamada de entropia cruzada binária:

```text
Log loss = - media[y*log(p) + (1-y)*log(1-p)]
```

Quanto maior a probabilidade atribuída à classe correta, menor a perda. Uma previsão errada e excessivamente confiante recebe uma penalidade alta.

### Por que regularizar?

A regularização controla a magnitude dos coeficientes para reduzir sobreajuste, principalmente quando há:

- muitas variáveis;
- multicolinearidade;
- poucos dados em relação ao número de atributos;
- separação quase perfeita entre as classes.

| Técnica | Efeito principal |
|---|---|
| L1 — Lasso | Pode zerar coeficientes e fazer seleção de variáveis |
| L2 — Ridge | Encolhe os coeficientes sem normalmente zerá-los |
| Elastic Net | Combina os efeitos de L1 e L2 |

Regularização forte demais pode provocar **underfitting**. Regularização fraca demais pode permitir **overfitting** e coeficientes instáveis.

---

## 5. Premissas e preparação dos dados

A regressão logística não exige normalidade dos atributos nem variância constante dos resíduos. As premissas e cuidados mais relevantes são:

1. **Observações independentes:** registros repetidos ou dependentes exigem tratamento adequado.
2. **Relação linear com o logit:** atributos contínuos devem ter relação aproximadamente linear com o logaritmo da chance, não necessariamente com a probabilidade.
3. **Ausência de multicolinearidade extrema:** atributos muito redundantes tornam os coeficientes instáveis.
4. **Quantidade suficiente de dados e eventos:** poucos exemplos da classe rara prejudicam as estimativas.
5. **Ausência de separação perfeita:** se uma variável separa completamente as classes, os coeficientes podem crescer sem limite.
6. **Dados de entrada numéricos:** variáveis categóricas precisam ser codificadas.

### É obrigatório padronizar?

Não para a definição matemática do modelo. Entretanto, padronizar costuma ser recomendado quando:

- há escalas muito diferentes;
- existe regularização;
- o otimizador apresenta convergência lenta;
- deseja-se comparar magnitudes dos coeficientes com mais cuidado.

Use `StandardScaler` como ponto de partida. Com outliers relevantes, avalie `RobustScaler` e investigue a origem desses valores.

> O scaler deve ser ajustado somente nos dados de treino, preferencialmente dentro de um pipeline, para evitar vazamento de dados.

### Variáveis categóricas

A teoria trabalha com entradas numéricas. Categorias podem ser usadas após codificação, normalmente one-hot encoding. É importante evitar uma codificação ordinal artificial quando não existe ordem real.

---

## 6. Pontos fortes, limitações e correções

| Ponto forte | Consequência |
|---|---|
| Treino e inferência rápidos | Bom baseline e adequado a bases grandes |
| Saída probabilística | Permite ajustar o limiar conforme o negócio |
| Coeficientes interpretáveis | Facilita explicar direção e intensidade dos efeitos |
| Regularização disponível | Ajuda a controlar complexidade e multicolinearidade |
| Fronteira de decisão simples | Tende a generalizar bem quando a relação é adequada |

| Limitação | Como lidar |
|---|---|
| Fronteira linear | Criar interações, transformações ou usar modelo não linear |
| Multicolinearidade | Remover redundâncias, regularizar ou reduzir dimensionalidade |
| Outliers e pontos influentes | Investigar, corrigir erros, transformar ou usar técnicas robustas |
| Classes desbalanceadas | Ajustar pesos, amostragem, limiar e métricas |
| Separação perfeita | Regularizar, rever variáveis e aumentar a amostra |
| Coeficientes difíceis de comparar | Padronizar e interpretar `exp(beta)` com contexto |
| Probabilidades mal calibradas | Avaliar curva de calibração, Brier score e calibradores |

### Desbalanceamento de classes

A regressão logística pode favorecer a classe majoritária. Acurácia isolada pode esconder esse problema.

Medidas possíveis:

- usar `class_weight="balanced"` ou pesos definidos pelo negócio;
- realizar oversampling ou undersampling apenas no treino;
- usar validação estratificada;
- ajustar o limiar de decisão;
- avaliar precisão, recall, F1, PR-AUC e matriz de confusão;
- escolher a métrica de acordo com o custo do erro.

Em fraude, por exemplo, reduzir falsos negativos pode ser mais importante que maximizar a acurácia.

---

## 7. Classificação multiclasse

A formulação original é binária, mas a regressão logística pode ser estendida para múltiplas classes.

### One-vs-Rest — OvR

Treina um classificador para cada classe contra todas as demais.

```text
Quantidade de classificadores = numero de classes
```

### Multinomial ou softmax

Modela todas as classes conjuntamente e produz probabilidades cuja soma é 1. Em geral, é a formulação natural para classes mutuamente exclusivas.

### One-vs-One — OvO

Treina um classificador para cada par de classes:

```text
Quantidade = C*(C - 1)/2
```

Pode transformar classificadores binários em multiclasse, mas não é a estratégia típica da regressão logística no scikit-learn.

> “One-vs-kfold” não é estratégia multiclasse. K-fold é uma técnica de validação.

---

## 8. Custo computacional

Não existe uma única complexidade universal: ela depende do solver, do número de registros `N`, atributos `P`, classes, iterações e regularização.

Uma iteração que percorre a base costuma custar aproximadamente:

```text
O(N * P)
```

Se forem necessárias `I` iterações:

```text
aproximadamente O(I * N * P)
```

Alguns solvers de segunda ordem manipulam matrizes relacionadas aos atributos e podem ter custo e memória maiores quando `P` cresce.

Depois do treinamento, uma predição binária custa aproximadamente:

```text
O(P) por registro
```

Logo, a resposta “custo linear” é uma aproximação didática válida para uma passagem pelos dados, mas não descreve todo processo de otimização.

---

## 9. Pegadinhas essenciais

1. **Regressão logística é um classificador**, apesar do nome.
2. Ela prevê uma probabilidade; a classe depende do **limiar**.
3. O limiar `0,5` não é obrigatório nem sempre é o melhor.
4. A função sigmoide não torna qualquer relação não linear automaticamente modelável: a fronteira continua linear nos atributos originais.
5. O modelo é linear no **logit**, não diretamente em `p`.
6. Coeficiente positivo aumenta a log-chance, não a probabilidade por uma quantidade fixa.
7. `exp(beta)` é uma razão de chances, não uma variação direta em pontos percentuais.
8. Coeficiente grande não prova causalidade nem importância absoluta.
9. A magnitude só é comparável quando escala e codificação são consideradas.
10. L1 pode zerar coeficientes; L2 geralmente apenas os reduz.
11. No scikit-learn, `C` é o inverso da regularização.
12. Padronização não é premissa teórica obrigatória, mas ajuda otimização e regularização.
13. Outliers podem afetar significativamente os coeficientes.
14. Acurácia pode ser enganosa em classes desbalanceadas.
15. One-vs-Rest e softmax são estratégias multiclasse; k-fold é validação.
16. Ajustar scaler, encoder ou amostragem antes da separação dos dados causa leakage.

---

## 10. Roteiro para responder na sabatina

Se perguntarem “como funciona?”, uma boa resposta é:

> Primeiro o modelo calcula uma combinação linear dos atributos. Essa combinação representa o logaritmo da chance do evento. A função sigmoide converte o resultado em probabilidade. Durante o treino, os coeficientes são ajustados por máxima verossimilhança, normalmente com regularização. Na inferência, a probabilidade é comparada a um limiar definido conforme o custo de negócio.

Se perguntarem “quando usar?”, responda:

> Quando preciso de um baseline rápido, interpretável e probabilístico, e quando uma fronteira aproximadamente linear é razoável. É especialmente útil quando explicabilidade, ajuste de limiar e estabilidade operacional são importantes.

Se perguntarem “quando não usar?”, responda:

> Quando o fenômeno exige relações muito não lineares que não podem ser representadas por transformações simples, ou quando interações complexas dominam o problema. Mesmo assim, compararia qualquer modelo mais complexo contra a regressão logística como baseline.

---

## 11. Checklist final

- [ ] Sei diferenciar probabilidade, odds e logit.
- [ ] Consigo explicar sigmoide e ponto de corte.
- [ ] Sei distinguir coeficientes aprendidos de hiperparâmetros.
- [ ] Entendo máxima verossimilhança e log loss conceitualmente.
- [ ] Sei diferenciar L1, L2 e Elastic Net.
- [ ] Lembro que `C` pequeno significa regularização forte.
- [ ] Sei explicar premissas, multicolinearidade e separação perfeita.
- [ ] Sei tratar outliers, escala e variáveis categóricas.
- [ ] Sei lidar com desbalanceamento sem depender de acurácia.
- [ ] Sei explicar OvR, softmax e OvO.
- [ ] Sei discutir custo de treino e inferência sem prometer uma complexidade única.
- [ ] Consigo relacionar limiar e erros ao impacto de negócio.

---

## Resumo de bolso

> A regressão logística é um classificador supervisionado e paramétrico que modela o logit como uma combinação linear dos atributos e transforma o resultado em probabilidade com a sigmoide. Seus parâmetros são os coeficientes e o intercepto; regularização, solver e `C` são hiperparâmetros. É rápida, interpretável e útil como baseline, mas exige cuidado com linearidade no logit, multicolinearidade, outliers, separação perfeita, desbalanceamento, escala e vazamento de dados. A decisão final depende de um limiar que deve refletir o custo dos erros do negócio.
