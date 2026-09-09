# Metricas de Avaliação

## COMO E QUANDO USAR CADA MÉTRICA DE AVALIAÇÃO - CASOS PRÁTICOS EM BANCO

| Métrica | Quando Usar | Por Que Usar | Caso Prático em Banco |
|---------|-------------|--------------|----------------------|
| **Acurácia** | Classes balanceadas | Medida geral de performance | Aprovação de cartões (taxa similar de aprovados/negados) |
| **Precisão** | Custo alto de falsos positivos | Evitar aprovar clientes inadimplentes | Prevenção de fraudes (evitar bloquear transações legítimas) |
| **Recall** | Custo alto de falsos negativos | Evitar deixar fraudes passarem | Detecção de lavagem de dinheiro (capturar todas as transações suspeitas) |
| **F1-Score** | Balancear precisão e recall | Quando ambas métricas são importantes | Scoring de crédito (equilíbrio entre risco e oportunidade) |
| **AUC-ROC** | Comparar modelos globalmente | Avaliar performance em todos os thresholds | Seleção do melhor modelo de crédito |
| **Matriz de Confusão** | Análise detalhada dos erros | Entender tipos específicos de erro | Diagnóstico de performance do modelo |

### FORMAS MAIS FÁCEIS DE INTERPRETAR AS MÉTRICAS DE AVALIAÇÃO

Imagine que vc tem um dataset com 100 dados de transação, 99 são normais e uma é fraude. Se vc escrever um código só para printar 'não é fraude' você vai ter uma acurácia de 99%, porque ela mostra o quanto vc tá prevendo correto em relação a todas tentativas, mas sabemos que isso não faz sentido, então o dado sofrerá com o desbalanceamento

Precisão é a taxa de positivos corretos, queremos diminuir ao máximo os falsos positivos, exemplo: classificar se o vídeo é apropriado para criança

Recall, taxa de negativos corretos, exemplo, falar que uma pessoa não tem cancer, quando na verdade ela tem

f1 score: equilibra precisão e recall

## O que é AUC e o que significa um AUC alto?
AUC é a área sob a curva ROC e mede a capacidade do modelo de separar as classes. A curva ROC compara a taxa de verdadeiros positivos com a taxa de falsos positivos para diferentes thresholds.

Um AUC alto significa que o modelo consegue diferenciar bem as classes, ou seja, ele tende a dar scores maiores para exemplos positivos do que para negativos.


## Por que AUC é melhor que accuracy em muitos casos?
AUC pode ser melhor que accuracy porque não depende de um threshold fixo de classificação.

Enquanto a accuracy avalia apenas a predição final, o AUC avalia a capacidade do modelo de separar as classes para todos os thresholds possíveis.

Além disso, a accuracy pode ser enganosa em datasets desbalanceados, enquanto o AUC continua sendo uma boa métrica para avaliar o poder de discriminação do modelo.

## Qual é a diferença entre ROC-AUC e PR-AUC (Precision-Recall AUC)? Em um cenário tão desbalanceado, por que a PR-AUC pode ser mais informativa que a ROC-AUC?
ROC-AUC avalia a relação entre taxa de verdadeiro positivo e taxa de falso positivo ao variar o threshold. Já PR-AUC avalia Precision e Recall. Em bases muito desbalanceadas, a ROC-AUC pode parecer boa porque a enorme quantidade de negativos torna o FPR relativamente pequeno. A PR-AUC tende a ser mais informativa porque mostra diretamente a capacidade do modelo de encontrar a classe positiva mantendo uma Precision aceitável.

## O que é KS e por que ele é muito usado em crédito?
KS é uma métrica que mede a máxima diferença entre a distribuição acumulada das classes positivas e negativas.

Ele indica o quanto o modelo consegue separar as duas classes, sendo muito usado em crédito e risco, onde queremos saber se o modelo consegue diferenciar bons e maus pagadores.

Quanto maior o KS, melhor a capacidade de discriminação do modelo.


## Qual a diferença entre AUC e KS?
AUC mede a capacidade geral do modelo de separar as classes ao longo de todos os thresholds, enquanto o KS mede a maior separação entre as distribuições de positivos e negativos em um ponto específico.

O AUC avalia o ranking global do modelo, enquanto o KS é muito usado para encontrar o melhor ponto de corte, por isso é comum em modelos de crédito e risco.


## Qual a diferença entre Gini e Entropy na escolha do split?
Tanto o Gini quanto a Entropy são métricas de impureza usadas para escolher a melhor divisão em uma árvore de decisão.

O Gini mede a probabilidade de classificar incorretamente um elemento se ele for rotulado aleatoriamente de acordo com a distribuição das classes.

Já a Entropy vem da teoria da informação e mede o nível de desordem do conjunto, sendo baseada em logaritmo. A Entropy tende a ser mais custosa computacionalmente por usar log, enquanto o Gini é mais simples de calcular, por isso o Gini é mais utilizado na prática.

Em geral, os dois produzem resultados muito semelhantes, mas a Entropy pode gerar divisões um pouco mais balanceadas em alguns casos.


## Qual a diferença entre Precision e Recall e quando usar cada uma?
Precision mede quantos dos positivos previstos pelo modelo realmente são positivos, enquanto Recall mede quantos dos positivos reais foram corretamente identificados pelo modelo.

Precision é importante quando o custo de falso positivo é alto, enquanto Recall é importante quando o custo de falso negativo é alto. Por exemplo, em detecção de fraude priorizamos recall, e em filtro de spam priorizamos precision.


## Quando usar F1-score ao invés de Precision ou Recall?
O F1-score é a média harmônica entre precision e recall, e é usado quando queremos um equilíbrio entre os dois. Ele é útil principalmente em datasets desbalanceados, onde a accuracy pode ser enganosa, e quando não podemos priorizar apenas precision ou apenas recall.

Como usa média harmônica, o F1-score só será alto se precision e recall forem altos ao mesmo tempo.


## Por que accuracy pode ser uma métrica ruim em dados desbalanceados?
Accuracy mede a proporção total de acertos, mas em datasets desbalanceados ela pode ser enganosa. Por exemplo, se 99% dos dados são da classe 0, um modelo que sempre prevê 0 terá 99% de accuracy, mesmo sem identificar nenhum caso da classe minoritária.

Nesses casos, métricas como precision, recall, F1 ou AUC são mais adequadas, porque avaliam melhor o desempenho na classe de interesse.


# Explique:
1.	O que é AUC
2.	O que é curva ROC
3.	O que significa um AUC alto
4.	Por que AUC é melhor que accuracy em problemas desbalanceados
5.	O que é Gini e qual a relação com AUC

A AUC é uma métrica usada em problemas de classificação para avaliar a capacidade do modelo de separar as classes.
Ela vem da curva ROC, que mostra a relação entre a taxa de verdadeiros positivos e a taxa de falsos positivos para diferentes thresholds.

A curva ROC é construída variando o threshold do modelo, e para cada valor calculamos o TPR, que é o recall, e o FPR, que é a taxa de falsos positivos.

A AUC é a área embaixo da curva ROC, e ela mede o quanto o modelo consegue distinguir as duas classes.
Quanto mais perto de 1, melhor o modelo está separando as classes.
Um AUC de 0.5 significa que o modelo é aleatório, e um AUC igual a 1 significa que ele separa perfeitamente.

A AUC é melhor que accuracy em problemas desbalanceados porque a accuracy pode ser alta mesmo quando o modelo não aprende nada.
Por exemplo, se eu tenho 95% de não fraude e 5% de fraude, um modelo que sempre prevê não fraude vai ter 95% de accuracy, mas não serve para nada.
A AUC não depende de um threshold fixo, então ela avalia o modelo de forma mais geral.

O Gini é uma transformação da AUC, muito usada em crédito.
A relação é:

Gini = 2 × AUC − 1


## Por que AUC é independente de threshold e acurácia não é?
A acurácia depende de threshold porque é calculada a partir das classes previstas, que são definidas usando um limite de probabilidade, normalmente 0.5.

Já a AUC é independente de threshold porque avalia o desempenho do modelo considerando todos os possíveis valores de threshold através da curva ROC, medindo a capacidade do modelo de separar as classes independentemente do ponto de corte.


## Quando KS pode ser mais útil que AUC?
KS pode ser mais útil que AUC quando precisamos avaliar a capacidade do modelo de separar as classes e escolher um ponto de corte para decisão.

Enquanto a AUC mede o desempenho geral do modelo em todos os thresholds, o KS mostra a maior distância entre as distribuições das classes, sendo muito utilizado em modelos de score de crédito para definir o cutoff ideal.

## Em um modelo de detecção de fraude, quando você priorizaria recall e qual seria a consequência dessa escolha?
Quando priorizar: quando deixar uma fraude passar — falso negativo — custa mais do que investigar uma transação legítima.
Consequência: para aumentar o recall, geralmente reduzimos o threshold. Assim, capturamos mais fraudes, mas também aumentamos os falsos positivos e, frequentemente, reduzimos a precision.

Eu priorizaria recall quando o custo de não detectar uma fraude for muito alto. Assim, busco capturar a maior proporção possível das fraudes reais. A consequência é que, normalmente, teremos mais falsos positivos, podendo bloquear transações legítimas e aumentar o custo de análise. Por isso, o threshold deve ser definido considerando o custo financeiro dos dois tipos de erro

## Se diminuirmos o threshold de classificação de 0,5 para 0,3, o que tende a acontecer com recall e precision? Por quê?
Recall aumenta: mais positivos reais são capturados e temos menos falsos negativos.
Precision tende a diminuir: junto com os novos positivos verdadeiros, entram mais falsos positivos.

Exemplo: uma transação com probabilidade de fraude de 0,4 seria considerada legítima com threshold 0,5, mas fraude com threshold 0,3.
Threshold menor → mais positivos previstos → recall maior e precision menor.

## Em detecção de câncer, o que representa um falso negativo e por que normalmente queremos minimizá-lo?
Um falso negativo ocorre quando um paciente doente é classificado como saudável. Em um exame de triagem, normalmente buscamos minimizá-lo porque ele pode atrasar o diagnóstico e o tratamento. Isso significa priorizar recall ou sensibilidade, embora o threshold deva considerar também os danos provocados pelos falsos positivos

## Em qual situação bancária você priorizaria precision em vez de recall? Explique usando falsos positivos.
Em um banco, priorizaríamos precision quando o custo de um falso positivo for muito alto.
Exemplo: um modelo que bloqueia automaticamente contas por suspeita de fraude.
- Positivo previsto: conta considerada fraudulenta.
- Falso positivo: conta legítima classificada como fraudulenta.
- Consequência: bloqueio indevido, cliente sem acesso ao dinheiro, reclamações e dano reputacional.

Eu priorizaria precision quando uma previsão positiva provocar uma ação severa, como o bloqueio automático de uma conta. Uma precision baixa geraria muitos falsos positivos, bloqueando clientes legítimos. Portanto, eu exigiria maior confiança antes de tomar essa ação.

## Qual é a diferença entre MAE e RMSE? Qual das duas é mais sensível a erros grandes/outliers e em que tipo de situação você escolheria uma em vez da outra?
MAE é a média dos erros absolutos e trata os erros de forma linear, sendo mais robusta a outliers. RMSE é a raiz da média dos erros ao quadrado e penaliza mais fortemente erros grandes, sendo mais sensível a valores extremos. Eu usaria MAE quando quero representar o erro típico de forma mais robusta e RMSE quando erros grandes têm custo maior