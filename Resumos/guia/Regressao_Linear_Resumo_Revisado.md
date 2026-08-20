# Regressão Linear, Ridge, Lasso e Elastic Net — Resumo para Sabatina

Este material organiza os conceitos centrais das imagens em formato de revisão. O objetivo não é reproduzir cada pergunta, mas preparar respostas curtas, corretas e justificadas.

---

## 1. Ideia central

A regressão é uma tarefa de aprendizado **supervisionado** usada para prever uma variável-alvo numérica, geralmente contínua.

Na regressão linear, estimamos uma reta, um plano ou, com várias variáveis, um **hiperplano** que represente a relação entre as entradas e a resposta.

```text
Classificação -> prevê uma classe ou categoria
Regressão     -> prevê um valor numérico contínuo
```

### Regressão simples e múltipla

```text
Regressão simples:  ŷ = β₀ + β₁x

Regressão múltipla: ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ
```

- `ŷ`: valor previsto.
- `β₀`: intercepto.
- `βⱼ`: coeficiente associado à variável `xⱼ`.
- `xⱼ`: variável independente, atributo ou feature.
- `y`: variável dependente, resposta ou target.

> **Pegadinha:** “linear” significa linear nos coeficientes. Uma regressão com `x²` continua sendo um modelo linear nos parâmetros, embora descreva uma curva em relação a `x`.

---

## 2. Como o modelo aprende

Para cada observação, o resíduo é a diferença entre o valor real e o previsto:

```text
eᵢ = yᵢ - ŷᵢ
```

O método dos mínimos quadrados ordinários, ou OLS, escolhe os coeficientes que minimizam a soma dos resíduos ao quadrado:

```text
SSE = Σ(yᵢ - ŷᵢ)²

MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

Elevar o erro ao quadrado evita o cancelamento entre erros positivos e negativos e penaliza fortemente erros grandes. Por isso, OLS é sensível a outliers.

Na formulação matricial, quando a solução existe de forma adequada:

```text
β̂ = (XᵀX)⁻¹Xᵀy
```

Na prática, bibliotecas usam métodos numericamente mais estáveis do que calcular a inversa diretamente.

---

## 3. Como interpretar os coeficientes

Em uma regressão múltipla, `βⱼ` representa a variação esperada em `y` quando `xⱼ` aumenta uma unidade, **mantendo as demais variáveis constantes**.

```text
βⱼ > 0 -> associação positiva com a resposta
βⱼ < 0 -> associação negativa com a resposta
|βⱼ|   -> intensidade na unidade e escala daquela variável
```

Cuidados importantes:

- coeficientes em escalas diferentes não devem ser comparados diretamente;
- mesmo após padronização, magnitude não é uma medida universal de importância;
- multicolinearidade pode tornar sinais e magnitudes instáveis;
- associação não implica causalidade;
- a interpretação depende da codificação de variáveis categóricas e da presença de interações.

---

## 4. Premissas da regressão linear

As principais premissas do modelo clássico são:

| Premissa | Significado | Se for violada |
|---|---|---|
| Linearidade | A média condicional de `y` é representada linearmente pelos termos do modelo | Viés sistemático e resíduos com padrão |
| Exogeneidade | `E(erro | X) = 0` | Coeficientes podem ficar viesados |
| Independência dos erros | Os resíduos não dependem uns dos outros | Inferência e incerteza ficam incorretas |
| Homoscedasticidade | A variância dos erros é aproximadamente constante | Erros-padrão clássicos ficam inadequados |
| Ausência de multicolinearidade perfeita | Nenhuma feature é combinação linear exata das outras | Coeficientes não são identificáveis |

A normalidade dos resíduos é especialmente importante para testes, intervalos e inferência em amostras pequenas. Ela **não é condição necessária** para calcular previsões por OLS.

### Como verificar

- resíduos versus valores previstos: linearidade e homoscedasticidade;
- Q-Q plot: normalidade aproximada dos resíduos;
- VIF e matriz de correlação: multicolinearidade;
- resíduos ao longo do tempo: autocorrelação;
- validação fora da amostra: capacidade de generalização.

---

## 5. Preparação dos dados

### Escala

Para OLS sem regularização, padronizar não é obrigatório e não muda as previsões quando a transformação é aplicada corretamente. Ainda assim, pode ajudar na comparação de coeficientes e na estabilidade numérica.

Para Ridge, Lasso e Elastic Net, padronizar é uma prática essencial, pois a penalização depende da magnitude dos coeficientes e, portanto, da escala das features.

> Ajuste o transformador somente no treino e aplique-o à validação e ao teste. O ideal é usar um `Pipeline` para evitar vazamento de dados.

### Variáveis categóricas

Strings não entram diretamente no modelo. É necessário convertê-las em números, normalmente com one-hot encoding. Ordinal encoding só é adequado quando existe ordem real entre as categorias.

### Outliers

Outliers e pontos de alta alavancagem podem alterar fortemente a reta. Antes de removê-los:

1. verifique se são erros de coleta;
2. analise resíduos, leverage e distância de Cook;
3. considere transformação, winsorização justificada ou regressão robusta;
4. documente qualquer exclusão.

> Reescalar os dados não elimina o impacto estatístico de um outlier.

---

## 6. Regularização: por que usar

Regularização adiciona uma penalidade à função de perda para limitar a magnitude dos coeficientes. Isso reduz a flexibilidade do modelo, normalmente:

```text
aumenta o viés + reduz a variância -> pode melhorar a generalização
```

Ela não garante melhora no teste. Um valor inadequado pode causar underfitting.

### Parâmetros aprendidos x hiperparâmetros

| Tipo | Exemplos |
|---|---|
| Parâmetros aprendidos | `β₀`, `β₁`, ..., `βₚ` |
| Hiperparâmetros | `alpha`, `l1_ratio`, inclusão do intercepto e opções do otimizador |

Em geral, o intercepto não é regularizado pelas implementações usuais.

---

## 7. Ridge — regularização L2

Ridge adiciona o quadrado da norma L2 dos coeficientes:

```text
Objetivo Ridge = SSE + α Σβⱼ²
               = SSE + α ||β||₂²
```

Fatores como `1/n` ou `1/(2n)` podem aparecer conforme a biblioteca. Eles alteram a escala numérica de `alpha`, não a ideia da penalização.

### Efeito de `alpha`

```text
α = 0       -> regressão linear sem regularização
α pequeno   -> penalização fraca
α grande    -> coeficientes mais próximos de zero e maior risco de underfitting
```

### Quando Ridge é uma boa escolha

- features correlacionadas;
- muitos preditores com pequenos efeitos;
- coeficientes instáveis ou alta variância;
- desejo de manter todas as features no modelo.

### Limitação principal

Ridge reduz os coeficientes, mas normalmente **não os zera exatamente**. Portanto, não é um método direto de seleção de features.

---

## 8. Lasso — regularização L1

Lasso adiciona a norma L1 dos coeficientes:

```text
Objetivo Lasso = (1 / 2n) SSE + α Σ|βⱼ|
               = (1 / 2n) SSE + α ||β||₁
```

### Principal característica

Lasso pode levar alguns coeficientes exatamente a zero. Assim, faz regularização e seleção de features simultaneamente.

### Quando usar

- há muitas features e se espera uma solução esparsa;
- parte das variáveis provavelmente é irrelevante;
- interpretabilidade por redução do conjunto de variáveis é desejada.

### Pontos negativos

- diante de features muito correlacionadas, pode escolher uma e descartar outra de maneira instável;
- com `p > n`, a solução pode selecionar no máximo cerca de `n` variáveis sob condições usuais;
- zerar um coeficiente não prova que a variável não tenha relação com o alvo;
- `alpha` excessivo causa underfitting.

Escolha `alpha` por validação cruzada, por exemplo com `LassoCV`, dentro de um pipeline de pré-processamento.

---

## 9. Elastic Net — combinação L1 + L2

Elastic Net combina a esparsidade do Lasso com a estabilidade do Ridge:

```text
Objetivo = (1 / 2n) SSE
         + α · l1_ratio · ||β||₁
         + [α · (1 - l1_ratio) / 2] · ||β||₂²
```

```text
l1_ratio = 1 -> comportamento Lasso
l1_ratio = 0 -> comportamento Ridge
0 < l1_ratio < 1 -> combinação das duas penalidades
```

É especialmente útil quando há muitas features, necessidade de seleção e grupos de variáveis correlacionadas. `alpha` e `l1_ratio` devem ser ajustados conjuntamente por validação cruzada, usando `ElasticNetCV` ou busca de hiperparâmetros.

---

## 10. Comparação direta

| Modelo | Penalidade | Zera coeficientes? | Com features correlacionadas | Uso típico |
|---|---|---:|---|---|
| OLS | Nenhuma | Não | Pode ficar instável | Relação simples, baixa multicolinearidade |
| Ridge | L2 | Normalmente não | Distribui o efeito entre elas | Estabilidade e redução de variância |
| Lasso | L1 | Sim | Pode escolher uma arbitrariamente | Seleção de features e esparsidade |
| Elastic Net | L1 + L2 | Sim | Tende a preservar grupos melhor que Lasso | Seleção com features correlacionadas |

---

## 11. Custo computacional

Não existe uma única complexidade que sirva para todas as implementações.

- OLS denso, com `n` linhas e `p` features, costuma ter custo aproximado de `O(np² + p³)` em soluções algébricas clássicas.
- A predição custa aproximadamente `O(np)` para `n` novos registros.
- Ridge pode ser resolvido por métodos algébricos ou iterativos.
- Lasso e Elastic Net normalmente usam otimização iterativa; o custo cresce com `n`, `p`, número de iterações e quantidade de valores testados na validação cruzada.
- Lasso/Elastic Net podem ser mais custosos que um único ajuste OLS, principalmente ao buscar hiperparâmetros.

Quando `p` é grande, matrizes esparsas, escolha do solver e paralelização da validação fazem diferença.

---

## 12. Avaliação do modelo

| Métrica | Interpretação | Atenção |
|---|---|---|
| MAE | Erro absoluto médio | Menos sensível a erros extremos que MSE |
| MSE | Erro quadrático médio | Penaliza fortemente erros grandes |
| RMSE | Raiz do MSE | Fica na unidade do target |
| R² | Fração da variabilidade explicada em relação à média | Pode ser negativo no teste |
| R² ajustado | Penaliza a inclusão de variáveis no contexto inferencial | Não substitui validação fora da amostra |

O desempenho deve ser medido em validação ou teste. Um R² alto no treino não garante generalização nem causalidade.

---

## 13. Pontos positivos, negativos e mitigação

| Pontos positivos | Pontos negativos | Como lidar |
|---|---|---|
| Simples, rápido e interpretável | Não modela não linearidades automaticamente | Transformações, polinômios e interações validados |
| Predição barata | Sensível a outliers | Diagnóstico e regressão robusta |
| Boa baseline | Multicolinearidade desestabiliza coeficientes | Remover redundâncias, Ridge ou Elastic Net |
| Coeficientes permitem inferência sob premissas | Heteroscedasticidade prejudica erros-padrão | Transformações ou erros-padrão robustos |
| Regularização controla variância | Exige ajuste de hiperparâmetros e escala | Pipeline com validação cruzada |

---

## 14. Pegadinhas frequentes

1. Regressão linear prevê valor contínuo; regressão logística é classificador.
2. Desbalanceamento de classes não é uma propriedade natural de regressão linear.
3. OLS não exige padronização, mas modelos regularizados dependem da escala.
4. Lasso pode zerar coeficientes; Ridge normalmente apenas os reduz.
5. `alpha = 0` remove a penalização e recupera o problema sem regularização.
6. Regularização pode aumentar o erro de treino e ainda melhorar o teste.
7. Mais regularização não significa sempre melhor generalização.
8. Resíduo é `real - previsto`; não é simplesmente um registro “incorreto”.
9. Coeficiente grande pode refletir escala, não importância.
10. Feature zerada pelo Lasso não é prova de irrelevância, especialmente com colinearidade.
11. Heteroscedasticidade não é resolvida automaticamente por Ridge ou Lasso.
12. Um modelo pode ser linear nos parâmetros e usar termos polinomiais.

---

## 15. Resposta curta para a sabatina

> Regressão linear é um modelo supervisionado e paramétrico que estima uma combinação linear das features para prever um alvo contínuo, geralmente minimizando a soma dos resíduos ao quadrado. Suas premissas incluem linearidade da média condicional, exogeneidade, independência dos erros, homoscedasticidade e ausência de multicolinearidade perfeita. OLS é interpretável, mas sensível a outliers e colinearidade. Ridge usa penalidade L2 para estabilizar e reduzir coeficientes; Lasso usa L1 e pode zerá-los, fazendo seleção de features; Elastic Net combina as duas. Em modelos regularizados, padronizo as features e escolho os hiperparâmetros por validação cruzada dentro de um pipeline.

---

## 16. Checklist de revisão rápida

- [ ] Sei diferenciar regressão, classificação e regressão logística.
- [ ] Sei explicar `ŷ`, coeficiente, intercepto e resíduo.
- [ ] Sei listar e diagnosticar as principais premissas.
- [ ] Sei explicar por que outliers afetam mínimos quadrados.
- [ ] Sei interpretar um coeficiente mantendo as demais features constantes.
- [ ] Sei diferenciar OLS, Ridge, Lasso e Elastic Net.
- [ ] Sei explicar `alpha` e `l1_ratio`.
- [ ] Sei dizer por que regularização exige atenção à escala.
- [ ] Sei escolher métricas de regressão e avaliar fora da amostra.
- [ ] Sei justificar a escolha do modelo com base no problema, e não apenas na métrica.
