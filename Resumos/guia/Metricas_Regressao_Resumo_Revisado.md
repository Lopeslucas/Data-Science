# Métricas de Avaliação para Modelos de Regressão

## Visão rápida

Em regressão, o modelo prevê um valor numérico contínuo. A avaliação deve responder principalmente:

1. Qual é o tamanho típico dos erros?
2. Erros grandes devem receber uma penalização maior?
3. O modelo explica melhor os dados do que uma previsão constante pela média?
4. O desempenho se mantém em dados não vistos?

Não existe uma métrica universalmente melhor. A escolha depende da unidade do alvo, do custo dos erros, da presença de outliers e da necessidade de uma medida absoluta ou relativa.

---

## Notação essencial

- $y_i$: valor real da observação $i$.
- $\hat{y}_i$: valor previsto para a observação $i$.
- $\bar{y}$: média dos valores reais.
- $e_i = y_i - \hat{y}_i$: resíduo.
- $n$: número de observações.
- $p$: número de variáveis preditoras.

O resíduo tem sinal; as métricas normalmente usam seu valor absoluto ou seu quadrado para impedir que erros positivos e negativos se anulem.

---

## Métricas fundamentais

### MAE — Mean Absolute Error

$$
\operatorname{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

Representa o erro absoluto médio.

- Quanto menor, melhor.
- Possui a mesma unidade da variável-alvo.
- É mais robusto a outliers do que MSE e RMSE.
- Trata linearmente todos os erros: dobrar o erro dobra sua contribuição.

Use quando desejar uma interpretação direta do erro típico ou quando erros extremos não devam dominar a avaliação.

### MSE — Mean Squared Error

$$
\operatorname{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Representa o erro quadrático médio.

- Quanto menor, melhor.
- Sua unidade é o quadrado da unidade do alvo.
- Penaliza fortemente erros grandes.
- É sensível a outliers.
- É diferenciável, característica conveniente para otimização.

Use quando erros grandes forem especialmente indesejáveis ou quando a função de treinamento for baseada em mínimos quadrados.

### RMSE — Root Mean Squared Error

$$
\operatorname{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

É a raiz quadrada do MSE.

- Quanto menor, melhor.
- Retorna à mesma unidade da variável-alvo.
- Continua penalizando erros grandes mais intensamente que o MAE.
- Em um mesmo conjunto de erros, $\operatorname{RMSE} \geq \operatorname{MAE}$.

Use quando quiser a penalização quadrática, mas com resultado interpretável na escala original.

### MAPE — Mean Absolute Percentage Error

$$
\operatorname{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i-\hat{y}_i}{y_i}\right|
$$

Representa o erro percentual absoluto médio.

- Quanto menor, melhor.
- É independente da unidade e facilita comparações relativas.
- Fica indefinido quando $y_i=0$.
- Torna-se instável quando valores reais estão próximos de zero.
- Pode favorecer previsões abaixo do valor real e produzir percentuais enormes.

Use somente quando valores reais forem estritamente positivos e distantes de zero, e quando a interpretação percentual fizer sentido para o negócio.

### Coeficiente de determinação — $R^2$

$$
R^2 = 1 -
\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

Compara o modelo com o baseline que sempre prevê a média do alvo.

- Quanto maior, melhor.
- $R^2=1$: ajuste perfeito.
- $R^2=0$: desempenho equivalente ao baseline da média.
- $R^2<0$: desempenho pior que prever a média, algo possível especialmente em teste.
- Não possui unidade.

$R^2$ não é “percentual de acertos” e não informa, sozinho, o tamanho dos erros.

### $R^2$ ajustado

$$
R^2_{aj} = 1-(1-R^2)\frac{n-1}{n-p-1}
$$

Corrige o $R^2$ levando em conta a quantidade de preditores.

- Penaliza a inclusão de variáveis que pouco melhoram o ajuste.
- Pode diminuir quando uma variável irrelevante é adicionada.
- É útil para comparar modelos explicativos ajustados sobre os mesmos dados e a mesma variável-alvo.
- Não substitui avaliação em validação ou teste.

---

## Comparação direta

| Métrica | Melhor valor | Unidade | Outliers | Interpretação principal |
|---|---:|---|---|---|
| MAE | 0 | Mesma do alvo | Menor sensibilidade | Erro absoluto típico |
| MSE | 0 | Unidade ao quadrado | Alta sensibilidade | Penalização forte de erros grandes |
| RMSE | 0 | Mesma do alvo | Alta sensibilidade | Erro quadrático na escala original |
| MAPE | 0% | Percentual | Instável perto de zero | Erro relativo médio |
| $R^2$ | 1 | Sem unidade | Pode ser afetado | Ganho sobre o baseline da média |
| $R^2_{aj}$ | Próximo de 1 | Sem unidade | Pode ser afetado | Ajuste considerando a quantidade de variáveis |

Uma boa avaliação costuma combinar:

- MAE ou RMSE para quantificar o erro na escala do problema;
- $R^2$ para indicar o ganho em relação à média;
- análise dos resíduos para verificar padrões que uma métrica agregada esconde.

---

## Como escolher a métrica

### Prefira MAE quando

- o custo cresce aproximadamente de forma linear com o erro;
- deseja uma explicação simples para o negócio;
- existem outliers legítimos que não devem dominar a avaliação.

### Prefira RMSE quando

- erros grandes possuem custo desproporcional;
- quer destacar previsões muito ruins;
- precisa do resultado na unidade original do alvo.

### Prefira MAPE quando

- a comunicação em percentual é necessária;
- os valores reais são positivos e não se aproximam de zero;
- a comparação relativa entre escalas é relevante.

### Use $R^2$ como complemento quando

- deseja comparar o modelo com o baseline da média;
- os modelos usam o mesmo alvo e o mesmo conjunto de avaliação;
- ele é acompanhado por uma métrica de erro absoluto.

---

## Técnicas de validação

### Holdout

Divide os dados uma única vez em treino e teste, podendo também reservar validação.

- É o método mais rápido entre holdout, k-fold e leave-one-out.
- Tem baixo custo computacional.
- Sua estimativa pode variar bastante conforme a divisão sorteada.

É adequado para bases grandes e para uma avaliação inicial, desde que a separação represente o cenário real.

### K-fold cross-validation

Divide os dados em $k$ partes. Em cada rodada, uma parte valida o modelo e as outras $k-1$ partes o treinam. Ao final, agregam-se as métricas das $k$ rodadas.

- Reduz a dependência de uma única divisão.
- Ajuda na seleção de modelo e no ajuste de hiperparâmetros.
- Expõe o modelo a diferentes subconjuntos de treino e validação.
- Exige aproximadamente $k$ treinamentos por configuração.

Valores como $k=5$ ou $k=10$ são comuns, mas devem considerar tamanho da base e custo de treinamento.

### Repeated k-fold

Repete o k-fold com novas divisões aleatórias.

- Produz uma estimativa mais estável.
- Permite observar a variabilidade da métrica.
- Custa aproximadamente $r \times k$ treinamentos, para $r$ repetições.

### Leave-one-out — LOOCV

É um k-fold extremo em que $k=n$: uma observação valida e as demais treinam o modelo, repetindo o processo $n$ vezes.

- Aproveita quase todos os dados para treino em cada rodada.
- Possui custo muito alto.
- Não é indicado para conjuntos com milhões de registros.

### Out-of-time

Treina com observações do passado e avalia em um período futuro.

- É a escolha correta quando há dependência temporal.
- Simula como o modelo será usado em produção.
- Não deve embaralhar observações futuras com passadas.
- Ajuda a detectar degradação, mudança de comportamento e drift.

Out-of-time não é sinônimo de holdout aleatório, k-fold comum ou leave-one-out.

---

## Vazamento de dados: pegadinha crítica

O pré-processamento não deve ser ajustado separadamente em treino e teste.

O fluxo correto é:

1. Separar treino e teste.
2. Ajustar imputação, escala, seleção de variáveis e demais transformações somente no treino.
3. Aplicar ao teste os transformadores já ajustados no treino.
4. Ajustar hiperparâmetros usando apenas treino e validação, preferencialmente dentro de um pipeline com validação cruzada.
5. Usar o teste uma única vez para a estimativa final.

Calcular médias, desvios, categorias ou selecionar variáveis usando o conjunto completo causa vazamento e gera uma estimativa otimista.

---

## Custo computacional

Para $n$ observações, calcular MAE, MSE, RMSE, MAPE ou $R^2$ é, em geral, $O(n)$.

O maior custo normalmente vem da validação:

| Estratégia | Treinamentos por configuração | Custo relativo |
|---|---:|---|
| Holdout | 1 | Baixo |
| K-fold | $k$ | Médio |
| Repeated k-fold | $r \times k$ | Alto |
| LOOCV | $n$ | Muito alto |

Se houver busca com $h$ combinações de hiperparâmetros, o custo do k-fold pode chegar aproximadamente a $h \times k$ treinamentos.

---

## Pegadinhas frequentes

1. **Menor $R^2$ não é melhor.** Para erros, menor é melhor; para $R^2$, maior é melhor.
2. **$R^2$ pode ser negativo.** Isso significa que o modelo perdeu para o baseline da média.
3. **$R^2$ alto não garante erro pequeno.** Sempre o acompanhe de MAE ou RMSE.
4. **Adicionar variáveis não reduz necessariamente o $R^2$ comum.** Em treino, ele tende a permanecer igual ou aumentar; o ajustado pode diminuir.
5. **RMSE usa raiz quadrada, não raiz cúbica.**
6. **MSE não está na unidade original.** Sua unidade é elevada ao quadrado.
7. **MAE é mais robusto a outliers que MSE/RMSE**, mas não é imune a eles.
8. **MAPE falha com zero e fica instável perto de zero.**
9. **Não compare métricas absolutas entre alvos com escalas diferentes** sem contextualização.
10. **Cross-validation não escolhe automaticamente a métrica.** A métrica deve refletir o custo do problema.
11. **Cross-validation aumenta o custo**, pois treina o modelo várias vezes.
12. **Séries temporais não devem usar divisão aleatória comum.** Preserve a ordem temporal.

---

## Resposta curta para sabatina

> Avalio regressão combinando uma métrica de erro, como MAE ou RMSE, com $R^2$. O MAE é mais interpretável e menos sensível a outliers; o RMSE penaliza mais erros grandes. MAPE é útil para erro relativo, mas não funciona bem com valores reais próximos de zero. Valido o modelo com holdout ou k-fold conforme o tamanho e o custo da base; se houver tempo, uso divisão temporal. Todo pré-processamento e ajuste de hiperparâmetros acontece somente dentro dos dados de treino para evitar vazamento.

---

## Checklist de revisão

- Sei diferenciar MAE, MSE, RMSE e MAPE?
- Sei informar a unidade de cada métrica?
- Sei explicar por que MSE e RMSE são sensíveis a outliers?
- Sei interpretar $R^2$ igual a 1, 0 e menor que 0?
- Sei diferenciar $R^2$ de $R^2$ ajustado?
- Sei escolher entre holdout, k-fold, LOOCV e out-of-time?
- Sei explicar por que LOOCV é inviável em bases muito grandes?
- Sei montar um pipeline sem vazamento de dados?
- Consigo justificar a métrica em termos do custo do negócio?

Se todas as respostas forem “sim”, os principais tópicos cobrados nas imagens estão cobertos de forma consistente.
