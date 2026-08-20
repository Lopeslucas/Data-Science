# KNN — Classificação e Regressão

Este material reorganiza os conceitos cobrados em um guia contínuo. O objetivo é reunir o que costuma ser exigido em uma sabatina: funcionamento, premissas, parâmetros, hiperparâmetros, custo, limitações e pegadinhas.

## Visão rápida

| Aspecto | KNN |
|---|---|
| Aprendizado | Supervisionado |
| Tarefas | Classificação e regressão |
| Natureza | Preditivo, não paramétrico e baseado em instâncias |
| Ideia central | Pontos próximos tendem a possuir respostas semelhantes |
| Treinamento | Armazena os dados; quase não ajusta uma função explícita |
| Predição | Calcula distâncias, escolhe os `k` vizinhos e agrega suas respostas |
| Escala | Não é exigência formal, mas normalmente é essencial |
| Principal gargalo | Predição e memória em bases grandes |

## Como funciona

Para uma nova observação `x`:

1. Definem-se `k`, métrica de distância, pesos e algoritmo de busca.
2. Calcula-se a distância de `x` aos registros de treino.
3. Selecionam-se os `k` registros mais próximos.
4. Agregam-se as respostas desses vizinhos.

Na classificação, a agregação é uma votação de classes. Na regressão, ela produz um valor numérico.

## KNN para regressão

Com pesos uniformes, a previsão usual é a média dos alvos dos vizinhos:

```text
ŷ(x) = (1/k) · Σ yᵢ
```

Com pesos por distância:

```text
ŷ(x) = Σ(wᵢ · yᵢ) / Σwᵢ
wᵢ = 1 / (dᵢ + ε)^q
```

Assim, vizinhos mais próximos influenciam mais a previsão. A mediana também pode ser usada em uma implementação personalizada e é mais robusta a valores extremos, mas o `KNeighborsRegressor` padrão do scikit-learn usa média uniforme ou média ponderada.

### `weights`

| Valor | Comportamento |
|---|---|
| `uniform` | Todos os `k` vizinhos possuem o mesmo peso |
| `distance` | O peso é inversamente proporcional à distância |
| função | Permite definir uma ponderação personalizada |

## O papel de `k`

| Situação | Efeito |
|---|---|
| `k` próximo de 1 | Fronteira/curva irregular, baixo viés, alta variância e maior risco de overfitting |
| `k` intermediário | Compromisso entre ruído e suavização |
| `k` próximo de `N` | Previsão aproxima-se da média global, alto viés, baixa variância e underfitting |

Com `k = 1`, o erro de treino pode ficar muito baixo, mas o modelo se torna sensível a ruído e outliers. O melhor `k` deve ser escolhido por validação cruzada, usando uma métrica de regressão adequada.

## Distâncias

```text
Euclidiana: d(x,y) = √Σ(xⱼ - yⱼ)²
Manhattan:  d(x,y) = Σ|xⱼ - yⱼ|
Minkowski:  d(x,y) = (Σ|xⱼ - yⱼ|ᵖ)^(1/p)
```

- `p = 1`: Manhattan.
- `p = 2`: Euclidiana.
- Hamming: útil para atributos binários/categóricos devidamente codificados.
- Mahalanobis: considera correlação e escala, mas exige estimar/inverter uma matriz de covariância e é mais custosa.

Em muitas dimensões, as distâncias tendem a ficar parecidas: é a maldição da dimensionalidade. Seleção de variáveis, redução de dimensionalidade e mais dados podem ajudar.

## Preparação dos dados

O KNN não exige matematicamente a mesma escala, mas uma variável de grande magnitude pode dominar a distância. Por isso, normalmente se usa:

- `StandardScaler` para padronização;
- `MinMaxScaler` para intervalo limitado;
- `RobustScaler` quando há valores extremos;
- `MaxAbsScaler`, especialmente em dados esparsos.

O escalonador deve ser ajustado somente no treino, preferencialmente dentro de um `Pipeline`, para evitar vazamento de dados.

Variáveis categóricas precisam de codificação compatível com a distância escolhida. One-hot com Euclidiana nem sempre traduz bem a semelhança semântica.

## Outliers e ruído

Outliers podem alterar a vizinhança e, na regressão, contaminar a média local. Não devem ser removidos automaticamente.

Como lidar:

- investigar se são erro, evento raro ou informação legítima;
- usar `RobustScaler` quando apropriado;
- testar `weights="distance"`;
- evitar `k` excessivamente pequeno;
- comparar média com uma agregação robusta personalizada;
- validar todo tratamento dentro dos folds.

## Custo computacional

Para `n` exemplos de treino e `d` variáveis:

| Etapa | Brute force |
|---|---|
| Treino | Aproximadamente `O(nd)` para armazenar os dados |
| Distâncias de uma consulta | `O(nd)` |
| Seleção completa por ordenação | até `O(n log n)` após as distâncias |
| Predição de `m` consultas | aproximadamente `O(mnd)` |
| Memória | `O(nd)` |

Quando se prediz cada ponto de uma base de tamanho semelhante ao treino, o custo global pode ser tratado como quadrático em `n`. Por isso, dizer apenas “o KNN é quadrático” é uma simplificação do cenário completo.

`KDTree` e `BallTree` podem acelerar buscas em baixa ou média dimensão. Em alta dimensão, perdem eficiência e o brute force pode ser competitivo. `leaf_size` troca custo de construção, busca e memória.

## Parâmetros aprendidos x hiperparâmetros

O KNN não estima um conjunto fixo de coeficientes. Seu estado aprendido é essencialmente a base de treino armazenada.

Principais hiperparâmetros:

- `n_neighbors`: quantidade de vizinhos;
- `weights`: uniforme, distância ou função personalizada;
- `metric` e `p`: definição de proximidade;
- `algorithm`: `auto`, `brute`, `kd_tree` ou `ball_tree`;
- `leaf_size`: configuração das árvores de busca.

## Pontos positivos

- simples de explicar e implementar;
- naturalmente não linear;
- serve para classificação e regressão;
- quase não possui custo de ajuste;
- adapta-se a padrões locais complexos.

## Pontos negativos e mitigação

| Limitação | Como mitigar |
|---|---|
| Predição lenta | KDTree/BallTree, aproximação, redução de dados ou outro modelo |
| Alto consumo de memória | seleção de amostras/variáveis e métodos aproximados |
| Sensível à escala | pipeline com escalonamento |
| Sensível a `k` e à métrica | validação cruzada |
| Sofre em alta dimensão | seleção de features, PCA ou modelo alternativo |
| Extrapola mal | usar apenas quando novos pontos estiverem cobertos pelo domínio de treino |

## Pegadinhas de sabatina

- KNN não é clustering só porque utiliza distâncias.
- É supervisionado, preditivo, não paramétrico e baseado em instâncias.
- “Treino barato” não significa modelo barato: o custo foi deslocado para a predição.
- Na regressão, a saída é média ou média ponderada dos alvos, não votação de classe.
- `weights="distance"` não escolhe automaticamente outro `k`; apenas muda a influência dos vizinhos escolhidos.
- Escalonar não é uma obrigação formal, mas ignorar escala costuma distorcer a vizinhança.
- O KNN interpola localmente e não é uma boa escolha para extrapolação.

## Resposta oral em 30 segundos

> KNN é um algoritmo supervisionado, não paramétrico e baseado em instâncias. Para prever um ponto novo, calcula sua distância aos dados de treino, seleciona os `k` vizinhos mais próximos e agrega suas respostas. Em classificação usa votação; em regressão, média ou média ponderada pela distância. `k` pequeno reduz o viés e aumenta a variância; `k` grande suaviza o modelo e pode causar underfitting. Como depende de distância, escala, dimensionalidade, outliers e custo de predição são seus principais cuidados.
