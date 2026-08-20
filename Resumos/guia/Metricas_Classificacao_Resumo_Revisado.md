# Métricas de classificação e validação — resumo revisado

## 1. Ideia central

Uma boa avaliação não pergunta apenas “quantos o modelo acertou?”, mas também:

- qual classe é importante;
- qual erro custa mais;
- se as classes estão desbalanceadas;
- se o modelo produz classes, scores ou probabilidades;
- como o limiar altera as decisões;
- se a validação representa o uso em produção.

## 2. Matriz de confusão binária

Adotando a classe positiva como referência:

| Real \ Predito | Positivo | Negativo |
|---|---:|---:|
| Positivo | TP | FN |
| Negativo | FP | TN |

- **TP:** positivo real previsto como positivo.
- **TN:** negativo real previsto como negativo.
- **FP:** negativo real previsto como positivo.
- **FN:** positivo real previsto como negativo.

> Sempre confira a orientação dos eixos. Bibliotecas e figuras podem apresentar linhas/colunas de maneira diferente.

## 3. Totais importantes

```text
Total = TP + TN + FP + FN

Positivos reais = TP + FN
Negativos reais = TN + FP

Positivos preditos = TP + FP
Negativos preditos = TN + FN
```

## 4. Métricas fundamentais

### Acurácia

Percentual total de acertos:

```text
acurácia = (TP + TN) / (TP + TN + FP + FN)
```

É útil quando as classes e os custos dos erros são razoavelmente equilibrados. Pode ser enganosa em bases desbalanceadas.

### Precisão da classe positiva

Entre os casos previstos como positivos, quantos são realmente positivos?

```text
precisão = TP / (TP + FP)
```

Priorize quando falso positivo é caro.

### Revocação, recall, sensibilidade ou TPR

Entre os positivos reais, quantos foram encontrados?

```text
recall = TP / (TP + FN)
```

Priorize quando falso negativo é caro.

### Especificidade ou TNR

Entre os negativos reais, quantos foram reconhecidos?

```text
especificidade = TN / (TN + FP)
```

### Taxa de falso positivo

```text
FPR = FP / (FP + TN) = 1 - especificidade
```

### F1-score

Média harmônica entre precisão e recall:

```text
F1 = 2 × precisão × recall / (precisão + recall)

F1 = 2TP / (2TP + FP + FN)
```

F1 é útil quando se busca equilíbrio entre FP e FN, mas ignora TN.

## 5. Exemplo calculado

Considere:

```text
TP = 23   FN = 7
FP = 10   TN = 60
```

```text
Total        = 100
Acurácia     = (23 + 60) / 100 = 0,83
Precisão     = 23 / (23 + 10)  ≈ 0,6969
Recall       = 23 / (23 + 7)   ≈ 0,7667
Especificidade = 60 / (60 + 10) ≈ 0,8571
F1           = 46 / (46 + 10 + 7) ≈ 0,7302
```

Se a instrução disser “não arredonde e use duas casas”, normalmente espera-se truncamento conforme as alternativas, mas em contexto profissional deve-se declarar explicitamente a regra numérica.

## 6. A classe de referência muda as métricas

Para avaliar a classe negativa como se fosse a positiva:

```text
TP_neg = TN
FP_neg = FN
FN_neg = FP
TN_neg = TP
```

Então:

```text
precisão_neg = TN / (TN + FN)
recall_neg   = TN / (TN + FP)
```

> “Precisão da classe negativa” não é sinônimo de especificidade. A primeira condiciona ao predito negativo; a segunda, ao negativo real.

## 7. Limiar de classificação

Modelos probabilísticos produzem um score ou probabilidade. A classe é obtida usando um limiar:

```text
se P(y=1 | x) ≥ limiar → classe 1
caso contrário         → classe 0
```

`0,5` é comum, mas não obrigatório.

- reduzir o limiar tende a aumentar recall e falsos positivos;
- elevar o limiar tende a aumentar precisão e falsos negativos.

O limiar deve ser escolhido na validação segundo custos, restrições e capacidade operacional — nunca no teste final.

## 8. ROC e AUC

A curva ROC mostra, para vários limiares:

```text
eixo Y: TPR = recall
eixo X: FPR
```

ROC-AUC mede a capacidade de ranquear um positivo acima de um negativo.

- `1,0`: separação perfeita;
- `0,5`: desempenho semelhante ao acaso no ranqueamento;
- abaixo de `0,5`: ranking possivelmente invertido.

ROC-AUC não escolhe o limiar e pode parecer otimista com classes muito desbalanceadas.

## 9. Curva Precision-Recall

A curva PR mostra precisão versus recall em vários limiares. PR-AUC ou Average Precision costuma ser mais informativa quando a classe positiva é rara e importante.

Seu baseline depende da prevalência da classe positiva; por isso, valores não devem ser comparados sem considerar a distribuição do alvo.

## 10. Probabilidade e calibração

Discriminação e calibração são conceitos diferentes:

- **discriminação:** ordenar positivos acima de negativos;
- **calibração:** probabilidades previstas coincidirem com frequências observadas.

Métricas úteis para probabilidades:

### Log loss binária

```text
log_loss = -(1/N) × Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]
```

Penaliza fortemente previsões erradas e muito confiantes.

### Brier score

```text
Brier = (1/N) × Σ (p_i - y_i)²
```

Quanto menor, melhor.

## 11. Qual métrica escolher

| Situação | Métrica inicial |
|---|---|
| Classes equilibradas e erros semelhantes | Acurácia + matriz de confusão |
| Falso positivo caro | Precisão, especificidade e custo |
| Falso negativo caro | Recall/sensibilidade e custo |
| Positivo raro | PR-AUC, precisão e recall |
| Equilíbrio entre precisão e recall | F1 ou Fβ |
| Qualidade de ranking | ROC-AUC e/ou PR-AUC |
| Probabilidade usada em decisão | Log loss, Brier e calibração |

Nenhuma métrica substitui a matriz de confusão no limiar operacional.

## 12. Multiclasse

Uma matriz de confusão pode ter qualquer número de classes. Para cada classe, usa-se uma visão um-contra-restante.

Formas de agregação:

- **macro:** média simples entre classes; trata todas igualmente;
- **weighted:** média ponderada pelo suporte de cada classe;
- **micro:** agrega contagens globais antes de calcular a métrica.

> Em classificação multiclasse de rótulo único, micro-F1 pode coincidir com acurácia. Isso não vale para todo cenário multilabel.

## 13. Holdout

Divide os dados uma vez em treino e teste ou treino/validação/teste.

Vantagens:

- simples;
- rápido;
- adequado para bases muito grandes.

Limitação: a estimativa depende mais da divisão aleatória.

## 14. K-Fold e Stratified K-Fold

No K-Fold:

1. divide-se o conjunto em `K` partes;
2. treina-se em `K-1` partes;
3. valida-se na parte restante;
4. repete-se `K` vezes.

Cada exemplo participa uma vez da validação. O custo é aproximadamente `K` treinamentos por configuração.

**Stratified K-Fold** busca preservar a proporção de classes em cada fold e é a escolha comum para classificação desbalanceada.

## 15. Leave-One-Out

É K-Fold com `K = N`: cada observação é usada uma vez como validação.

- usa quase todos os dados em cada treino;
- exige `N` treinamentos;
- tem custo muito alto;
- pode produzir estimativa de alta variância.

Em bases grandes, raramente é a melhor escolha prática.

## 16. Validação temporal, por grupos e repetida

### Temporal / out-of-time

Treina no passado e valida no futuro. Não se deve embaralhar séries temporais como se fossem amostras independentes.

### Group K-Fold

Mantém registros do mesmo cliente, paciente, equipamento ou entidade no mesmo fold, evitando vazamento entre entidades.

### Repeated K-Fold

Repete o K-Fold com diferentes partições para estimar melhor a variabilidade, ao custo de mais treinamentos.

### Validação aninhada

Usa um laço interno para ajustar hiperparâmetros e um externo para estimar desempenho. É útil quando se deseja uma comparação menos otimista entre modelos.

## 17. Vazamento de dados

O pré-processamento deve ser aprendido apenas com o treino de cada fold:

- imputação;
- scaler;
- seleção de variáveis;
- PCA;
- oversampling/SMOTE;
- escolha de limiar;
- ajuste de hiperparâmetros.

Use pipeline. O conjunto de teste final não participa de nenhuma decisão.

> Ajustar um scaler separadamente no teste também está errado. O correto é ajustar no treino e apenas transformar validação/teste.

## 18. Custo da validação

Se um treino custa `T`:

```text
holdout             ≈ 1 × T por configuração
K-Fold              ≈ K × T por configuração
Repeated K-Fold     ≈ repetições × K × T
Leave-One-Out       ≈ N × T
Nested CV           ≈ folds externos × busca interna × T
```

## 19. Pegadinhas de sabatina

- Acurácia pode ser a pior métrica em um conjunto altamente desbalanceado.
- Precisão usa positivos **preditos** no denominador.
- Recall usa positivos **reais** no denominador.
- F1 não usa TN.
- ROC é uma curva; AUC é a área sob essa curva.
- AUC mede ranking, não calibração nem desempenho em um limiar específico.
- Matriz de confusão funciona em multiclasse.
- “Melhor modelo” depende do custo do erro e do limiar operacional.
- Estratificação preserva proporções de classes; não acelera o algoritmo.
- Holdout é mais rápido; Leave-One-Out é normalmente o mais caro.
- Dados temporais e agrupados exigem divisões específicas.

## 20. Resposta curta para entrevista

> Eu começo pela matriz de confusão e pelo custo de falso positivo e falso negativo. Em dados equilibrados, acurácia pode servir; com positivo raro, observo precisão, recall, F1 e PR-AUC. ROC-AUC avalia ranking, não o limiar nem a calibração. Escolho métrica e limiar na validação, uso divisão estratificada, temporal ou por grupos conforme o problema e mantenho todo pré-processamento dentro de pipeline para evitar vazamento.

## 21. Checklist

- [ ] A classe positiva está explicitamente definida?
- [ ] O custo de FP e FN foi discutido?
- [ ] A distribuição das classes foi examinada?
- [ ] A métrica corresponde ao objetivo de negócio?
- [ ] O limiar foi escolhido apenas na validação?
- [ ] A matriz de confusão no limiar final foi inspecionada?
- [ ] A divisão respeita tempo, grupos e estratificação?
- [ ] Todo pré-processamento está dentro do pipeline?
- [ ] O teste final permaneceu intocado?
- [ ] Foram reportadas média e dispersão entre folds quando aplicável?
