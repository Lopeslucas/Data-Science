# Guia de domínio — Métricas de Avaliação e Validação para Classificação

Material consolidado a partir das perguntas sobre **matriz de confusão, acurácia, precisão, recall, especificidade, F1, ROC/AUC, classificação multiclasse, limiar e técnicas de validação**.

Uma resposta segura deve seguir:

> **objetivo do problema → classe positiva → custo dos erros → métrica → limiar → validação sem vazamento**

---

# Parte I — Antes de calcular qualquer métrica

## 1.1 Defina a classe positiva

“Positivo” não significa bom. É apenas a classe tratada como evento de interesse.

Exemplos:

- fraude;
- doença;
- falha de equipamento;
- cancelamento;
- cliente que responderá a uma campanha.

Trocar a classe positiva troca TP, FP, FN, TN e as métricas por classe.

## 1.2 Defina o custo dos erros

- **Falso positivo:** gera alarme/intervenção quando o evento não ocorreu.
- **Falso negativo:** deixa de detectar um evento real.

Não existe uma melhor métrica universal. A escolha depende do custo relativo desses erros, prevalência e uso operacional.

## 1.3 Score, probabilidade, limiar e classe

Muitos classificadores produzem score ou probabilidade. Um limiar `t` transforma isso em classe:

\[
\hat y=1\quad\text{se}\quad score\geq t
\]

`t=0,5` é comum quando há probabilidades, não obrigatório. Alterar o limiar muda a matriz de confusão e métricas dependentes dela.

---

# Parte II — Matriz de confusão

## 2.1 O que é?

É uma tabela cruzada entre classes reais e classes preditas. Cada célula conta quantos registros possuem determinada combinação.

Ela pode ser usada em classificação binária e multiclasse.

## 2.2 Convenção das linhas e colunas

Não existe uma orientação visual universal. Algumas matrizes usam:

- linhas = real, colunas = predito;
- linhas = predito, colunas = real.

Antes de interpretar, leia os eixos. Decorar “TP fica no canto superior esquerdo” é perigoso.

Neste guia, quando não indicado o contrário:

| | Predito positivo | Predito negativo |
|---|---:|---:|
| **Real positivo** | TP | FN |
| **Real negativo** | FP | TN |

## 2.3 As quatro contagens

- **TP — verdadeiro positivo:** real positivo, predito positivo.
- **TN — verdadeiro negativo:** real negativo, predito negativo.
- **FP — falso positivo:** real negativo, predito positivo.
- **FN — falso negativo:** real positivo, predito negativo.

Macete: o primeiro termo diz se a previsão acertou; o segundo diz o que foi previsto.

## 2.4 Totais importantes

\[
N=TP+TN+FP+FN
\]

\[
RealPositivo=TP+FN
\]

\[
RealNegativo=TN+FP
\]

\[
PreditoPositivo=TP+FP
\]

\[
PreditoNegativo=TN+FN
\]

## 2.5 Exemplo das imagens

Com `TP=23`, `FN=7`, `FP=10`, `TN=60`:

- total = `100`;
- reais positivos = `30`;
- reais negativos = `70`;
- preditos positivos = `33`;
- preditos negativos = `67`.

---

# Parte III — Métricas binárias fundamentais

## 3.1 Acurácia

Proporção total de acertos:

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

No exemplo:

\[
Accuracy=\frac{23+60}{100}=0,83
\]

### Quando é útil

- classes razoavelmente equilibradas;
- custos de FP e FN semelhantes;
- baseline majoritário considerado.

### Limitação

Com 99% de negativos, prever sempre negativo produz 99% de acurácia e recall positivo zero.

## 3.2 Precisão — Precision/PPV

Entre os registros preditos como positivos, quantos realmente são positivos?

\[
Precision=\frac{TP}{TP+FP}
\]

No exemplo:

\[
Precision_+=\frac{23}{33}\approx0,6969
\]

Alta precisão é importante quando falsos positivos são caros.

## 3.3 Recall — Sensibilidade/TPR

Entre os positivos reais, quantos foram encontrados?

\[
Recall=Sensitivity=TPR=\frac{TP}{TP+FN}
\]

No exemplo:

\[
Recall_+=\frac{23}{30}\approx0,7667
\]

Alto recall é importante quando falsos negativos são caros.

## 3.4 Especificidade — TNR

Entre os negativos reais, quantos foram corretamente rejeitados?

\[
Specificity=TNR=\frac{TN}{TN+FP}
\]

No exemplo:

\[
Specificity=\frac{60}{70}\approx0,8571
\]

Especificidade é o recall da classe negativa quando a negativa é tratada como alvo.

## 3.5 Taxa de falso positivo — FPR

\[
FPR=\frac{FP}{FP+TN}=1-Specificity
\]

## 3.6 Taxa de falso negativo — FNR

\[
FNR=\frac{FN}{FN+TP}=1-Recall
\]

## 3.7 Valor preditivo negativo — NPV

Entre os preditos negativos, quantos realmente são negativos?

\[
NPV=\frac{TN}{TN+FN}
\]

## 3.8 F1-score

Média harmônica entre precisão e recall:

\[
F1=2\frac{Precision\cdot Recall}{Precision+Recall}
\]

Forma equivalente:

\[
F1=\frac{2TP}{2TP+FP+FN}
\]

No exemplo:

\[
F1_+=\frac{46}{46+10+7}=\frac{46}{63}\approx0,7302
\]

F1 ignora TN. É útil quando o foco está na classe positiva, mas não representa todos os custos possíveis.

## 3.9 F-beta

\[
F_\beta=(1+\beta^2)\frac{Precision\cdot Recall}{\beta^2Precision+Recall}
\]

- `β>1`: dá mais peso ao recall;
- `β<1`: dá mais peso à precisão;
- `β=1`: F1.

## 3.10 Balanced accuracy

\[
BalancedAccuracy=\frac{Recall+Specificity}{2}
\]

É mais informativa que acurácia em muitos cenários desbalanceados.

## 3.11 MCC

Matthews Correlation Coefficient:

\[
MCC=\frac{TP\cdot TN-FP\cdot FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
\]

Considera as quatro células e é útil em classificação binária desbalanceada.

## 3.12 Métricas da classe negativa

Ao tratar “negativa” como classe de interesse:

\[
Precision_- = \frac{TN}{TN+FN}
\]

\[
Recall_- = \frac{TN}{TN+FP}
\]

\[
F1_- = 2\frac{Precision_-\cdot Recall_-}{Precision_-+Recall_-}
\]

No exemplo:

- `Precision_- = 60/67 ≈ 0,8955`;
- `Recall_- = 60/70 ≈ 0,8571`;
- `F1_- ≈ 0,8759`.

---

# Parte IV — Arredondamento e cálculo seguro

## 4.1 Não arredonde valores intermediários

Calcule usando contagens ou precisão completa e arredonde apenas o resultado final.

## 4.2 “Não arredonde e use duas casas”

Essa instrução normalmente significa truncar, mas é ambígua. Por exemplo, `0,6969`:

- arredondado: `0,70`;
- truncado: `0,69`.

Em análise real, declare explicitamente a regra. Não trunque silenciosamente.

## 4.3 Divisão por zero

Uma métrica pode ficar indefinida quando o denominador é zero, por exemplo se nenhum positivo for predito. Bibliotecas podem retornar 0, `NaN` ou emitir warning conforme configuração.

---

# Parte V — Curvas dependentes do limiar

## 5.1 Curva ROC

Varia o limiar e plota:

- eixo Y: TPR/recall;
- eixo X: FPR.

Cada ponto representa uma política de decisão diferente.

## 5.2 ROC-AUC

É a área sob a curva ROC. Pode ser interpretada como a probabilidade de um positivo escolhido ao acaso receber score maior que um negativo escolhido ao acaso, sob condições usuais de tratamento de empates.

- `1,0`: ranking perfeito;
- `0,5`: ranking equivalente ao acaso;
- `<0,5`: ranking invertido ou problema de orientação.

AUC mede ranking, não calibração e não escolhe limiar operacional.

## 5.3 Curva Precision-Recall

Plota precisão contra recall ao variar o limiar. É especialmente útil quando a classe positiva é rara e o interesse está nela.

## 5.4 PR-AUC e Average Precision

Resumem o trade-off precisão-recall. Average Precision e área trapezoidal sob PR não são sempre numericamente idênticas.

A linha de base da precisão depende aproximadamente da prevalência positiva, ao contrário da linha de base 0,5 da ROC-AUC.

## 5.5 ROC-AUC versus PR-AUC

- ROC-AUC avalia ranking global entre positivos e negativos.
- PR-AUC enfatiza desempenho na classe positiva.
- Com muitos negativos, ROC-AUC pode parecer boa enquanto a precisão operacional é baixa.

Use ambas quando agregarem informação, mas escolha a principal pelo problema.

## 5.6 Escolha do limiar

O limiar deve ser escolhido na validação com base em:

- custo de FP e FN;
- recall ou precisão mínimos;
- capacidade operacional;
- F-beta;
- lucro/utilidade esperada;
- restrições de equidade e segurança.

Não escolha limiar usando o teste final.

---

# Parte VI — Métricas probabilísticas e calibração

## 6.1 Log loss

No binário:

\[
LogLoss=-\frac{1}{N}\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]
\]

Penaliza fortemente previsões erradas e confiantes.

## 6.2 Brier score

\[
Brier=\frac{1}{N}\sum_i(p_i-y_i)^2
\]

Avalia erro quadrático probabilístico.

## 6.3 Calibração

Se previsões próximas de 0,8 realmente ocorrem cerca de 80% das vezes, o modelo está calibrado nessa região.

Avalie com:

- curva de calibração;
- Brier score;
- log loss;
- calibração por subgrupos e faixas;
- métodos como Platt ou isotonic ajustados sem vazamento.

Bom AUC não garante calibração; boa calibração não garante bom ranking.

---

# Parte VII — Multiclasse

## 7.1 Matriz C×C

Para C classes, a célula `(i,j)` conta exemplos reais da classe i preditos como j, de acordo com a orientação adotada.

- diagonal: acertos;
- fora da diagonal: confusões específicas.

## 7.2 One-vs-Rest para cada classe

Para avaliar a classe c:

- `TP_c`: real c, predito c;
- `FN_c`: real c, predito outra;
- `FP_c`: real outra, predito c;
- `TN_c`: todas as demais combinações.

Então aplicam-se as fórmulas binárias.

## 7.3 Exemplo das imagens

Matriz com linhas reais e colunas preditas:

| Real \ Predito | Elephant | Monkey | Fish | Lion |
|---|---:|---:|---:|---:|
| Elephant | 25 | 3 | 0 | 2 |
| Monkey | 3 | 53 | 2 | 3 |
| Fish | 2 | 1 | 24 | 2 |
| Lion | 1 | 0 | 2 | 71 |

Totais:

- N = `194`;
- reais Fish = `29`;
- preditos Fish = `28`;
- acertos = `173`;
- acurácia = `173/194 ≈ 0,8918`.

Para Fish:

- `TP=24`;
- `FN=2+1+2=5`;
- `FP=0+2+2=4`;
- recall = `24/29 ≈ 0,8276`;
- precisão = `24/28 ≈ 0,8571`;
- F1 = `48/(48+4+5) ≈ 0,8421`.

## 7.4 Macro, micro e weighted

### Macro

Calcula a métrica por classe e faz média simples. Dá o mesmo peso a todas as classes.

### Weighted

Faz média por classe ponderada pelo suporte real. Classes frequentes influenciam mais.

### Micro

Soma TP, FP e FN de todas as classes antes de calcular. Em classificação single-label, micro precision, micro recall e micro F1 coincidem com a acurácia.

## 7.5 Support

Support é o número de exemplos reais de cada classe. Não é uma métrica de qualidade, mas ajuda a interpretar estabilidade e médias.

## 7.6 Cuidado com “métricas da classe negativa”

Em multiclasse, “negativo” para uma classe significa a união de todas as outras. O TN pode ser enorme e mascarar problemas. Prefira métricas por classe e macro quando todas importam.

---

# Parte VIII — Comparando modelos

## 8.1 “Qual modelo é melhor?”

Não pode ser respondido sem objetivo. Um modelo pode ter:

- maior acurácia;
- menor recall positivo;
- maior precisão;
- pior calibração;
- custo operacional maior.

Escolha segundo a função de utilidade do problema.

## 8.2 Exemplo A versus B das imagens

Modelo A: `TP=23`, `FN=7`, `FP=10`, `TN=60`.

Modelo B: `TP=23`, `FN=7`, `FP=58`, `TN=12`.

Ambos têm o mesmo recall positivo:

\[
23/(23+7)=0,7667
\]

Mas A possui muito menos FP, maior precisão positiva, especificidade, acurácia e F1. Se os dados e limiares são comparáveis, A domina B nas métricas relevantes mostradas.

## 8.3 Exemplo desbalanceado das imagens

Com `TN=3302`, `FP=1335`, `FN=65`, `TP=256`, acurácia é influenciada pela grande classe negativa. A pergunta “qual a pior métrica?” é mal formulada: acurácia tende a ser a menos informativa isoladamente, mas nenhuma métrica deve ser escolhida apenas olhando a matriz sem custo de negócio.

## 8.4 Compare incerteza, não apenas ponto estimado

Considere:

- intervalos bootstrap;
- variação entre folds;
- testes pareados apropriados;
- tamanho do efeito;
- estabilidade por tempo e subgrupos.

Diferença de 0,01 pode ser ruído ou importante, dependendo do volume e custo.

---

# Parte IX — Holdout

## 9.1 Como funciona

Divide os dados uma vez, por exemplo:

- treino;
- validação;
- teste.

Ou treino/teste quando a validação ocorre internamente.

## 9.2 Pontos positivos

- rápido;
- simples;
- adequado para bases grandes;
- avaliação direta em conjunto não visto.

## 9.3 Pontos negativos

- estimativa depende da divisão aleatória;
- usa menos dados no treino;
- pode gerar proporções ruins;
- pequena base produz alta variância.

Mitigue com estratificação, grupos/tempo adequados ou validação repetida.

---

# Parte X — K-Fold Cross-Validation

## 10.1 Como funciona

Divide os dados em K folds. Em cada rodada:

- um fold é validação;
- K−1 folds são treino;
- o processo ocorre K vezes.

Cada exemplo é usado uma vez para validação e K−1 vezes em treino.

## 10.2 Para que serve

- estimar generalização;
- comparar modelos;
- selecionar hiperparâmetros;
- avaliar estabilidade;
- aproveitar melhor bases pequenas/médias.

Ela não escolhe automaticamente a melhor métrica; a métrica vem do objetivo.

## 10.3 Custo

Treinar uma configuração custa aproximadamente K ajustes. Com H configurações, cerca de `K×H`, antes de repetições.

## 10.4 Escolha de K

- K pequeno: menor custo, possível maior viés na estimativa;
- K grande: mais treino por fold e maior custo; correlação entre treinos aumenta;
- 5 ou 10 são escolhas comuns, não regras universais.

---

# Parte XI — Stratified K-Fold

## 11.1 O que faz

Preserva aproximadamente a proporção de classes em cada fold.

Não garante proporções exatamente idênticas em todos os casos, especialmente com poucas amostras da classe rara.

## 11.2 Quando usar

- classificação desbalanceada;
- métricas instáveis quando uma classe falta em um fold;
- dados independentes sem restrição temporal/grupal superior.

Estratificação não resolve vazamento por paciente, cliente, máquina ou tempo.

---

# Parte XII — Leave-One-Out (LOOCV)

## 12.1 Como funciona

Com N registros, cria N folds:

- um exemplo para validação;
- N−1 para treino;
- repete N vezes.

## 12.2 Vantagens

- quase todos os dados são usados em cada treino;
- baixo viés da estimativa de treino em alguns contextos.

## 12.3 Limitações

- N treinos;
- muito caro para bases grandes;
- estimativa pode ter alta variância;
- folds altamente correlacionados;
- algumas métricas como AUC não são definidas em um único exemplo por fold, exigindo previsões agregadas.

Para um milhão de linhas, LOOCV normalmente é impraticável.

---

# Parte XIII — Validação temporal e “out-of-time”

## 13.1 Como funciona

Treina no passado e valida/testa em período futuro. Não é sinônimo de holdout aleatório.

## 13.2 Por que usar

- previsão de eventos futuros;
- drift e sazonalidade;
- evitar treinar com informação do futuro;
- reproduzir implantação real.

## 13.3 Variações

- janela expansiva;
- janela deslizante;
- gap/embargo entre treino e validação;
- múltiplos backtests.

Em dados temporais, embaralhar e estratificar aleatoriamente pode criar vazamento.

---

# Parte XIV — Grouped Cross-Validation

Quando várias linhas pertencem ao mesmo paciente, cliente, equipamento ou evento, todas devem ficar no mesmo lado da divisão.

Use:

- GroupKFold;
- StratifiedGroupKFold quando possível;
- GroupShuffleSplit;
- LeaveOneGroupOut.

Sem isso, o modelo pode “reconhecer” a entidade na validação.

---

# Parte XV — Repeated Cross-Validation

Repete K-fold com novas divisões aleatórias. Ajuda a estimar a variabilidade causada pelo particionamento.

Custo aproximado:

\[
K\times R
\]

treinos por configuração, onde R é o número de repetições.

Não significa repetir literalmente cada fold idêntico; cada repetição cria uma nova partição.

---

# Parte XVI — Nested Cross-Validation

Quando se ajustam muitos hiperparâmetros e se deseja estimativa menos otimista:

- loop interno: seleciona hiperparâmetros;
- loop externo: estima desempenho do processo de seleção.

É computacionalmente cara, mas evita avaliar no mesmo conjunto usado para escolher a configuração.

---

# Parte XVII — Vazamento de dados

## 17.1 Regra central

Qualquer transformação que aprende com os dados deve ser ajustada **somente no treino** de cada divisão:

- scaler;
- imputação;
- encoding aprendido;
- seleção de atributos;
- PCA;
- reamostragem;
- seleção de limiar;
- calibração;
- ajuste de hiperparâmetros.

## 17.2 Correção da pegadinha

Não se deve ajustar o pré-processamento “separadamente no treino e no teste”. Deve-se:

1. ajustar no treino;
2. aplicar a transformação aprendida à validação/teste.

Ajustar um scaler ou encoder novamente no teste cria representações inconsistentes e usa informação do teste.

## 17.3 Pipeline

Coloque pré-processamento e modelo em um pipeline para que cada fold ajuste tudo corretamente.

## 17.4 Teste final

O teste deve ser usado uma vez ao final. Consultá-lo repetidamente para escolher features, modelos ou limiar transforma-o em validação e torna a estimativa otimista.

---

# Parte XVIII — Como escolher a métrica

| Situação | Métricas prioritárias |
|---|---|
| Classes equilibradas, custos parecidos | Acurácia + matriz |
| FN muito caro | Recall, F-beta com β>1 |
| FP muito caro | Precisão, especificidade |
| Classe positiva rara | PR-AUC, precisão/recall, F1 |
| Ranking de casos | ROC-AUC e/ou PR-AUC |
| Probabilidade deve ser confiável | Log loss, Brier, calibração |
| Todas as classes importam igualmente | Macro recall/F1 |
| Desbalanceamento binário geral | Balanced accuracy, MCC |
| Decisão econômica | Custo/utilidade esperada |

Uma métrica nunca substitui inspeção da matriz, prevalência, limiar e subgrupos.

---

# Parte XIX — Pontos positivos e limitações das métricas

## 19.1 Acurácia

- positiva: intuitiva;
- negativa: enganosa com desbalanceamento e custos diferentes.

## 19.2 Precisão

- positiva: controla qualidade dos alertas positivos;
- negativa: depende da prevalência e não mede positivos perdidos.

## 19.3 Recall

- positiva: mede cobertura da classe positiva;
- negativa: pode ser maximizada prevendo tudo positivo.

## 19.4 F1

- positiva: equilibra precisão e recall;
- negativa: ignora TN, assume pesos simétricos e depende do limiar.

## 19.5 ROC-AUC

- positiva: independente de um limiar, avalia ranking;
- negativa: pode parecer otimista em classe rara e não mede calibração.

## 19.6 PR-AUC

- positiva: focada na classe positiva rara;
- negativa: depende da prevalência e comparações entre bases com prevalências diferentes exigem cuidado.

## 19.7 Log loss/Brier

- positiva: avaliam probabilidades;
- negativa: exigem scores probabilísticos e interpretação alinhada à calibração.

---

# Parte XX — Pegadinhas presentes nas questões

1. **Leia os eixos da matriz.** A posição de TP/FN/FP/TN pode mudar.
2. **Matriz de confusão funciona em multiclasse.**
3. **Positivo não significa “bom”.** É a classe-alvo.
4. **Precisão usa preditos positivos no denominador.**
5. **Recall usa positivos reais no denominador.**
6. **Especificidade é recall da classe negativa.**
7. **F1 é média harmônica, não aritmética.**
8. **F1 ignora TN.**
9. **Acurácia pode falhar em dados desbalanceados.**
10. **0,5 é limiar comum, não obrigatório.**
11. **ROC é a curva; AUC é a área.**
12. **ROC usa TPR contra FPR, não precisão contra recall.**
13. **AUC não mede calibração.**
14. **PR-AUC é frequentemente mais informativa para positivos raros.**
15. **Average Precision não é sempre igual à área trapezoidal PR.**
16. **Não existe melhor métrica sem custo do erro.**
17. **Macro dá igual peso às classes; weighted pondera por support.**
18. **Micro F1 coincide com acurácia em multiclasse single-label.**
19. **TN de uma classe multiclasse inclui todas as demais combinações corretas/erradas que não envolvem essa classe como real ou predita.**
20. **Arredondar e truncar são operações diferentes.**
21. **K-fold treina K vezes por configuração.**
22. **LOOCV é K-fold com K=N.**
23. **LOOCV não é automaticamente melhor por usar mais treino.**
24. **Holdout é rápido, mas mais dependente da divisão.**
25. **Estratificação preserva aproximadamente proporções; não resolve grupos ou tempo.**
26. **Out-of-time não é holdout aleatório.**
27. **Repeated K-fold recria partições; não repete folds idênticos.**
28. **Validação cruzada ajuda a escolher modelo/hiperparâmetro, não define a métrica de negócio.**
29. **Pré-processamento não deve ser ajustado no teste.** Ajuste no treino e apenas transforme o teste.
30. **Reamostragem deve ocorrer dentro de cada fold de treino.**
31. **Escolher limiar no teste é vazamento.**
32. **Comparar modelos exige mesmas divisões e incerteza pareada quando possível.**
33. **Métrica global pode esconder falhas em subgrupos.**
34. **Teste final reutilizado deixa de ser teste final.**

---

# Parte XXI — Respostas-modelo

## “O que é matriz de confusão?”

> É uma tabela de contagens entre classes reais e preditas. No binário ela contém TP, TN, FP e FN; no multiclasse é uma matriz C×C. Antes de interpretar, verifico qual eixo representa real e qual representa predito.

## “Qual a diferença entre precisão e recall?”

> Precisão pergunta quantos alertas positivos estavam corretos: TP/(TP+FP). Recall pergunta quantos positivos reais foram encontrados: TP/(TP+FN). Priorizo precisão quando FP é caro e recall quando FN é caro.

## “Quando usar F1?”

> Quando quero equilibrar precisão e recall e a classe positiva é central. F1 é a média harmônica e pune quando uma das duas é baixa, mas ignora TN e não substitui uma função de custo.

## “Por que acurácia pode enganar?”

> Porque uma classe majoritária domina o total. Um modelo que sempre prevê a maioria pode ter alta acurácia e recall zero para a classe rara. Comparo com baseline e uso matriz, recall, precisão, PR-AUC ou balanced accuracy.

## “ROC-AUC ou PR-AUC?”

> ROC-AUC mede ranking global por TPR e FPR. PR-AUC enfatiza precisão e recall da classe positiva e costuma ser mais informativa quando ela é rara. Nenhuma mede calibração nem escolhe o limiar operacional.

## “Como escolher o limiar?”

> Escolho na validação com base no custo de FP e FN, metas de precisão/recall e capacidade operacional. O limiar 0,5 é apenas um padrão. Depois congelo a regra e avalio no teste final.

## “Como avaliar multiclasse?”

> Analiso matriz C×C, métricas por classe e médias macro, weighted e micro. Macro é importante quando todas as classes têm igual valor; weighted reflete frequência; micro agrega decisões e pode esconder classes raras.

## “Para que serve validação cruzada?”

> Para estimar generalização, comparar configurações e ajustar hiperparâmetros em múltiplas divisões. Todo pré-processamento deve ser reajustado dentro de cada fold de treino. O teste final fica fora desse processo.

## “Holdout, K-fold ou LOOCV?”

> Holdout é mais rápido e adequado a bases grandes. K-fold equilibra custo e estabilidade em bases pequenas/médias. LOOCV treina N vezes, é caro e não é automaticamente mais estável. Também considero estrutura temporal e grupos antes de escolher.

## “O que é vazamento?”

> É quando informação que não estaria disponível no treino influencia o modelo ou a seleção. Inclui ajustar scaler, imputação, seleção, reamostragem, hiperparâmetro ou limiar usando validação/teste fora do pipeline correto.

---

# Parte XXII — Fluxo recomendado

1. defina população, unidade e classe positiva;
2. defina custo de FP/FN e métrica principal;
3. reserve teste final respeitando tempo/grupo;
4. escolha holdout ou cross-validation apropriada;
5. coloque pré-processamento em pipeline;
6. gere previsões out-of-fold para comparação;
7. avalie matriz, métrica principal e secundárias;
8. examine curvas de limiar e calibração;
9. escolha limiar na validação;
10. avalie subgrupos e intervalos de incerteza;
11. congele pipeline e regra;
12. execute teste final uma única vez;
13. monitore prevalência, drift e métricas em produção.

---

# Parte XXIII — Roteiro de treinamento

## Sessão 1 — Matriz binária

- identificar TP, TN, FP e FN em qualquer orientação;
- calcular totais reais e preditos;
- traduzir erros para o negócio.

## Sessão 2 — Métricas

- calcular acurácia, precisão, recall, especificidade e F1;
- comparar classe positiva e negativa;
- escolher métrica por custo.

## Sessão 3 — Limiar e curvas

- variar limiar;
- interpretar ROC e PR;
- distinguir ranking e calibração;
- escolher limiar operacional.

## Sessão 4 — Multiclasse

- transformar uma classe em one-vs-rest;
- calcular métricas por classe;
- comparar macro, micro e weighted.

## Sessão 5 — Validação

- comparar holdout, K-fold e LOOCV;
- aplicar estratificação, grupos e tempo;
- construir pipeline sem vazamento;
- explicar nested e repeated CV.

---

# Checklist final de domínio

Você domina o conteúdo quando consegue, sem consultar material:

- ler matriz em qualquer orientação;
- definir TP, TN, FP e FN;
- calcular totais reais/preditos;
- calcular e interpretar acurácia, precisão, recall, especificidade, F1, F-beta, balanced accuracy e MCC;
- explicar quando acurácia engana;
- distinguir ROC, ROC-AUC, PR e PR-AUC;
- distinguir ranking de calibração;
- escolher limiar por custo;
- calcular métricas por classe em multiclasse;
- explicar macro, micro, weighted e support;
- comparar modelos sem dizer “melhor” sem objetivo;
- explicar holdout, K-fold, estratificado, LOOCV, repetido e nested;
- respeitar tempo e grupos;
- colocar pré-processamento, reamostragem e seleção dentro dos folds;
- preservar teste final;
- reconhecer e corrigir todas as pegadinhas listadas.
