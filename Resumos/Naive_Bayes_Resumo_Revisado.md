# Naive Bayes — resumo revisado para sabatina

## 1. Visão geral

Naive Bayes é uma família de classificadores supervisionados e probabilísticos baseados no Teorema de Bayes. O termo *naive* vem da hipótese forte de que os atributos são condicionalmente independentes dada a classe.

É um modelo:

- preditivo e supervisionado;
- paramétrico em suas variantes usuais;
- naturalmente multiclasse;
- rápido no treino e na inferência;
- especialmente útil em texto, contagens e problemas de alta dimensão esparsa.

## 2. Teorema de Bayes

```text
P(C | X) = P(X | C) × P(C) / P(X)
```

Onde:

- `P(C | X)`: posterior — probabilidade da classe após observar os atributos;
- `P(X | C)`: verossimilhança — probabilidade dos atributos dada a classe;
- `P(C)`: prior — probabilidade da classe antes de observar `X`;
- `P(X)`: evidência — igual para todas as classes na mesma observação.

Para escolher a classe, basta maximizar:

```text
classe prevista = argmax_C P(X | C) × P(C)
```

## 3. Hipótese de independência

Se `X = (x₁, x₂, ..., xₚ)`, o modelo aproxima:

```text
P(X | C) = Π P(x_j | C)
```

Logo:

```text
P(C | X) ∝ P(C) × Π P(x_j | C)
```

A hipótese correta é **independência condicional à classe**, não independência absoluta entre os atributos.

Mesmo quando essa hipótese é violada, o classificador pode funcionar bem. Porém, atributos fortemente redundantes podem contar evidência semelhante várias vezes e produzir probabilidades excessivamente confiantes.

## 4. Variantes principais

### Gaussian Naive Bayes

Indicado para atributos contínuos, assumindo uma distribuição normal por atributo e classe.

Parâmetros aprendidos por classe:

```text
média μ_cj
variância σ²_cj
prior P(C=c)
```

### Multinomial Naive Bayes

Indicado para contagens ou frequências não negativas, como frequência de palavras.

Exemplos: classificação de documentos, spam e tópicos.

### Bernoulli Naive Bayes

Indicado para atributos binários, como presença ou ausência de uma palavra.

### Categorical Naive Bayes

Indicado para atributos categóricos codificados como categorias discretas.

> Pegadinha: a variante deve combinar com a distribuição/representação dos atributos. Não se escolhe Gaussian apenas porque o alvo é numérico — o alvo continua sendo uma classe.

## 5. Treinamento e predição

No treinamento, o modelo estima:

- frequência/prior das classes;
- probabilidades condicionais dos atributos;
- ou médias e variâncias, no caso gaussiano.

Na predição, calcula um escore posterior para cada classe e escolhe o maior.

Na prática, usam-se logaritmos:

```text
log score(C) = log P(C) + Σ log P(x_j | C)
```

Isso transforma produtos em somas e evita **underflow numérico** causado pela multiplicação de muitas probabilidades pequenas.

## 6. Probabilidade zero e suavização de Laplace

Se uma categoria nunca apareceu com determinada classe, sua probabilidade estimada pode ser zero. Como as probabilidades são multiplicadas, um único zero zera o escore da classe.

Suavização aditiva:

```text
P(x=v | C=c) = (contagem(v,c) + α) /
               (contagem(c) + α × número_de_valores)
```

- `α = 1`: suavização de Laplace;
- `0 < α < 1`: suavização de Lidstone;
- `α = 0`: sem suavização.

> Não confundir: logaritmos tratam underflow; suavização trata probabilidades estimadas iguais a zero.

## 7. Parâmetros e hiperparâmetros

### Parâmetros aprendidos

- priors das classes;
- probabilidades condicionais;
- médias e variâncias no GaussianNB.

### Hiperparâmetros comuns

- `alpha`: suavização em Multinomial/Bernoulli/Categorical;
- `fit_prior`: aprende ou não os priors das classes;
- `class_prior`: priors fornecidos manualmente;
- `var_smoothing`: estabilidade numérica do GaussianNB;
- `binarize`: limiar de binarização no BernoulliNB.

## 8. Escala, outliers e correlação

### Escala

Naive Bayes não usa distância. Em geral, padronizar não é obrigatório. A necessidade depende da variante e da preparação dos atributos, não de uma exigência geométrica como no KNN.

### Outliers

No GaussianNB, extremos podem distorcer médias e variâncias. Em variantes por contagem, valores anormalmente altos também podem dominar a evidência.

Antes de remover, investigue a origem. Possíveis respostas: transformação, discretização, winsorização justificada ou outro modelo de distribuição.

### Atributos correlacionados

Correlação forte viola a hipótese simplificadora e pode duplicar evidência. Seleção de variáveis, remoção de redundância ou outro modelo podem ajudar.

## 9. Classes desbalanceadas

Os priors refletem a frequência das classes. Assim, a classe majoritária pode receber vantagem inicial relevante.

Alternativas:

- priors definidos com conhecimento do domínio;
- reamostragem somente no treino;
- avaliação estratificada;
- escolha de limiar a partir do custo dos erros;
- métricas como recall, precisão, F1 e PR-AUC.

Não se deve concluir que alterar os priors sempre corrige o problema: a calibração e o custo de falso positivo/falso negativo precisam ser avaliados.

## 10. Custo computacional

Com `N` exemplos, `P` atributos e `C` classes:

- treino: aproximadamente `O(NP)`;
- predição por exemplo: aproximadamente `O(CP)`;
- memória: geralmente `O(CP)` mais estruturas de categorias.

É muito mais barato em inferência que um KNN por força bruta, pois não compara a nova observação com todo o treino.

## 11. Pontos positivos

- treino e inferência rápidos;
- baixa necessidade de memória;
- funciona bem com alta dimensão e dados esparsos;
- naturalmente multiclasse;
- bom baseline;
- suporta aprendizado incremental em algumas implementações;
- produz escores probabilísticos.

## 12. Pontos negativos e como reduzir

| Limitação | Como lidar |
|---|---|
| Independência raramente é exata | remover redundância ou comparar com outros modelos |
| Probabilidade zero | suavização aditiva |
| Underflow numérico | somar log-probabilidades |
| Probabilidades mal calibradas | validar calibração; Platt ou isotônica quando necessário |
| Sensibilidade a outliers no GaussianNB | tratamento robusto e diagnóstico |
| Distribuição inadequada | escolher a variante correta ou transformar os dados |
| Desbalanceamento | priors, reamostragem, limiar e métricas apropriadas |

## 13. Pegadinhas de sabatina

- Naive Bayes não é baseado em instâncias.
- É classificador, não algoritmo de regressão nas variantes abordadas.
- Pode resolver problemas binários e multiclasse.
- `P(A|B)` não é igual a `P(B|A)`.
- A evidência `P(X)` pode ser omitida somente ao comparar classes para a mesma observação.
- GaussianNB assume normalidade de cada atributo **condicionada à classe**.
- MultinomialNB espera atributos não negativos de contagem/frequência.
- BernoulliNB modela presença/ausência.
- Normalização não é obrigatória apenas porque há variáveis em escalas diferentes.
- Probabilidades produzidas podem não estar bem calibradas.

## 14. Resposta curta para entrevista

> Naive Bayes aplica o Teorema de Bayes e assume independência condicional dos atributos dada a classe. No treino estima priors e distribuições condicionais; na predição soma log-probabilidades e escolhe a classe de maior posterior. É rápido e forte para texto e alta dimensão, mas sofre com atributos redundantes, distribuição inadequada, probabilidade zero e possível má calibração.

## 15. Checklist

- [ ] A variante corresponde ao tipo dos atributos?
- [ ] Há atributos altamente redundantes?
- [ ] Foi aplicada suavização quando necessária?
- [ ] O cálculo usa log-probabilidades?
- [ ] Há outliers que distorcem parâmetros?
- [ ] Os priors são coerentes com produção?
- [ ] A validação é estratificada?
- [ ] Probabilidade e calibração foram avaliadas além da acurácia?
