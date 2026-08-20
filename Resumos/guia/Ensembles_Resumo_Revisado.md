# Ensembles — Bagging, Random Forest, Boosting e AdaBoost

Este material reorganiza os conceitos cobrados em um guia contínuo. O objetivo é reunir o que costuma ser exigido em uma sabatina: funcionamento, premissas, parâmetros, hiperparâmetros, custo, limitações e pegadinhas.


## 1. Visão geral

Um **ensemble** combina previsões de vários modelos para obter uma resposta final mais estável ou mais precisa que a de um estimador isolado.

É normalmente usado em aprendizado supervisionado para:

- classificação;
- regressão.

Principais famílias:

- **Voting/Averaging:** combina previsões diretamente;
- **Bagging:** treina modelos independentes em amostras diferentes;
- **Random Forest:** bagging de árvores com aleatoriedade adicional nas variáveis;
- **Boosting:** treina modelos sequencialmente, concentrando-se nos erros anteriores;
- **Stacking:** aprende um meta-modelo para combinar modelos base.

## 2. Por que combinar modelos funciona

O ganho depende de duas condições:

1. os modelos individuais precisam ter algum poder preditivo;
2. seus erros não devem ser perfeitamente correlacionados.

Muitos modelos iguais cometendo exatamente os mesmos erros não produzem diversidade útil.

Regra conceitual:

```text
ensemble forte = modelos competentes + erros suficientemente diversos
```

## 3. Viés e variância

- **Viés alto:** o modelo é simples demais e tende a underfitting.
- **Variância alta:** o resultado muda muito com pequenas mudanças no treino e tende a overfitting.

Em geral:

- Bagging reduz principalmente **variância**.
- Boosting reduz principalmente **viés**, embora também possa alterar a variância.

Essa é uma tendência, não uma lei absoluta.

## 4. Bagging

Bagging significa **Bootstrap Aggregating**.

Processo:

1. gerar várias amostras bootstrap do treino;
2. ajustar um modelo base em cada amostra;
3. combinar as previsões.

Classificação:

```text
ŷ = voto majoritário dos modelos
```

Regressão:

```text
ŷ = média das previsões
```

Os modelos são independentes entre si e podem ser treinados em paralelo.

## 5. Bootstrap e Out-of-Bag

Uma amostra bootstrap costuma ter o mesmo tamanho `N` do treino, mas é obtida **com reposição**. Portanto:

- alguns registros aparecem mais de uma vez;
- outros não aparecem.

Para `N` grande, a fração esperada de exemplos distintos em uma amostra é:

```text
1 - e^(-1) ≈ 63,2%
```

Os aproximadamente `36,8%` não escolhidos para uma árvore são seus exemplos **out-of-bag (OOB)**.

Predições OOB agregadas podem estimar generalização sem uma validação separada, embora não substituam sempre uma avaliação final independente.

> Pegadinha: 36,8% é aproximadamente a proporção que fica fora da amostra, não necessariamente a proporção de registros duplicados.

## 6. Quando Bagging ajuda

Funciona melhor com estimadores de alta variância e instáveis, como árvores profundas.

Se o modelo base tem viés muito alto, tirar a média de várias versões semelhantes pode manter o mesmo erro sistemático.

Pontos fortes:

- reduz variância;
- é paralelizável;
- melhora estabilidade;
- OOB oferece estimativa interna;
- costuma ser robusto quando há diversidade.

Limitações:

- mais memória e custo de inferência;
- interpretabilidade menor;
- modelos muito correlacionados reduzem o ganho;
- não resolve automaticamente desbalanceamento ou dados ruins.

## 7. Random Forest

Random Forest combina:

1. bootstrap das linhas;
2. árvores de decisão como modelos base;
3. subconjunto aleatório de variáveis considerado em cada divisão;
4. votação ou média das árvores.

A seleção aleatória de variáveis reduz correlação entre as árvores.

### Hiperparâmetros importantes

- `n_estimators`: número de árvores;
- `max_features`: variáveis consideradas por divisão;
- `max_depth`: profundidade máxima;
- `min_samples_split` e `min_samples_leaf`;
- `max_leaf_nodes`;
- `criterion`: Gini, entropia/log-loss ou critérios de regressão;
- `bootstrap` e `max_samples`;
- `class_weight`;
- `oob_score`;
- `n_jobs` e `random_state`.

### Pontos importantes

- não exige padronização para árvores;
- lida com não linearidades e interações;
- aceita classificação e regressão;
- mais árvores tendem a estabilizar o resultado, mas aumentam tempo e memória;
- aumentar `n_estimators` isoladamente normalmente não causa overfitting relevante, embora não corrija árvores mal configuradas.

## 8. Boosting

Boosting constrói uma sequência de modelos em que cada etapa tenta corrigir limitações do conjunto anterior.

Características:

- treinamento sequencial;
- modelos dependentes;
- modelos base frequentemente simples;
- combinação ponderada das previsões.

Ao contrário do bagging, não é naturalmente paralelizável entre etapas.

Boosting não é restrito a árvores, embora árvores rasas sejam muito comuns.

## 9. AdaBoost

No AdaBoost de classificação, todas as observações começam com o mesmo peso:

```text
w_i = 1 / N
```

A cada rodada:

1. treina-se um estimador usando os pesos atuais;
2. calcula-se seu erro ponderado;
3. atribui-se maior peso ao estimador mais competente;
4. aumentam-se os pesos dos exemplos errados;
5. normalizam-se os pesos e repete-se.

Para classificação binária, uma forma comum do peso do estimador é:

```text
α_t = 0,5 × ln((1 - erro_t) / erro_t)
```

Predição final:

```text
classe = sinal(Σ α_t × h_t(x))
```

Quanto menor o erro do estimador, maior seu peso. Se o erro se aproxima de `0,5` no caso binário, sua contribuição se aproxima de zero.

## 10. Bagging versus Boosting

| Aspecto | Bagging | Boosting |
|---|---|---|
| Treinamento | Independente | Sequencial |
| Paralelização | Natural | Limitada entre etapas |
| Dados | Amostras bootstrap | Pesos/resíduos mudam ao longo das etapas |
| Modelo base comum | Forte e instável | Fraco e simples |
| Foco típico | Reduzir variância | Reduzir viés |
| Sensibilidade a ruído | Menor | Pode ser maior |
| Combinação | Voto/média | Soma ponderada |

## 11. Parâmetros aprendidos e hiperparâmetros

### Parâmetros aprendidos

- regras e valores das árvores;
- pesos dos estimadores no boosting;
- distribuições de pesos/resíduos intermediários;
- parâmetros internos de cada modelo base.

### Hiperparâmetros comuns

- número de estimadores;
- complexidade do estimador base;
- taxa de aprendizado no boosting;
- proporção de linhas/variáveis amostradas;
- critérios de parada;
- regularização do modelo base;
- estratégia de votação ou meta-modelo.

> `n_estimators`, `max_depth` e `learning_rate` são hiperparâmetros, não parâmetros aprendidos.

## 12. Custo computacional

Se o custo de treinar um modelo base é `T` e há `B` estimadores:

```text
treino do bagging ≈ O(B × T)
```

O tempo de parede pode cair com paralelização, mas o trabalho total permanece proporcional ao número de modelos.

No boosting:

```text
treino ≈ soma do custo das B etapas sequenciais
```

Inferência em ambos:

```text
custo ≈ soma do custo de previsão dos B modelos
```

Para Random Forest, o custo depende de número de árvores, profundidade, amostras e variáveis candidatas. Não existe uma única classificação universal como “linear” sem declarar o que varia.

## 13. Dados problemáticos

### Desbalanceamento

Bagging e Random Forest ainda podem favorecer a maioria. Use `class_weight`, amostragem balanceada e métricas adequadas. AdaBoost também pode concentrar peso em casos difíceis, inclusive ruído.

### Outliers e rótulos errados

Árvores são relativamente robustas à escala, mas não imunes. Boosting pode perseguir rótulos errados porque aumenta o foco nos erros.

### Escala

A necessidade vem do modelo base. Ensembles de árvores não exigem padronização; ensembles contendo KNN ou modelos sensíveis à escala herdam essa necessidade.

## 14. Pegadinhas de sabatina

- Ensemble é estratégia de combinação, não um único algoritmo.
- Bagging, boosting e stacking são ensembles; leave-one-out e one-vs-rest não são.
- Bootstrap é amostragem com reposição.
- OOB existe por estimador e pode estimar o desempenho agregado.
- Random Forest usa o mesmo tipo de modelo base: árvores; a diversidade vem das amostras e variáveis.
- Random Forest não exige scaler.
- Mais árvores aumentam custo e estabilidade, mas não garantem ganho indefinido.
- Boosting não usa apenas árvores nem exige sempre profundidade 1.
- AdaBoost usa voto ponderado, não voto simples.
- Um ensemble não é automaticamente melhor: falta de diversidade pode anular o ganho.

## 15. Resposta curta para entrevista

> Ensembles combinam modelos para reduzir erros. Bagging treina modelos independentes em amostras bootstrap e reduz principalmente variância; Random Forest acrescenta seleção aleatória de variáveis para descorrelacionar árvores. Boosting treina modelos sequenciais para corrigir erros, reduzindo principalmente viés. O ganho depende da qualidade e diversidade dos modelos, com custo maior de treino, memória e inferência.

## 16. Checklist

- [ ] O problema é de viés, variância ou ambos?
- [ ] Os modelos base geram erros diversos?
- [ ] O custo de vários estimadores cabe em produção?
- [ ] Há ruído, outliers ou rótulos errados?
- [ ] O desbalanceamento foi tratado e medido corretamente?
- [ ] O pré-processamento adequado ao modelo base está em pipeline?
- [ ] Hiperparâmetros foram validados sem tocar no teste?
- [ ] O conjunto final foi avaliado em dados independentes?
