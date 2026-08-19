# Guia de domínio — Modelos de Machine Learning

Material consolidado a partir das perguntas das listas sobre **Classificação, KNN e Árvores de Decisão**.

## 1. Mapa do que precisa ser dominado

As perguntas avaliam cinco camadas de conhecimento:

1. **Enquadrar o problema:** classificação versus regressão; binário versus multiclasse.
2. **Explicar o modelo:** como aprende, como prediz e quais hipóteses faz.
3. **Configurar o algoritmo:** hiperparâmetros, métricas, pesos e critérios de parada.
4. **Diagnosticar erros:** overfitting, underfitting, outliers, escala e desbalanceamento.
5. **Avaliar custo e aplicação:** complexidade, estruturas de busca e escolhas práticas.

Uma resposta realmente completa deve percorrer: **definição → mecanismo → consequência → exemplo → limitação**.

---

# Parte I — Fundamentos de classificação

## O que é classificação?

Classificação é uma tarefa supervisionada cujo alvo é **categórico**. As variáveis de entrada podem ser numéricas, categóricas ou uma combinação das duas, desde que sejam tratadas de forma compatível com o modelo.

Exemplos:

- prever se um computador falhará: classe `falha` ou `não falha`;
- identificar se uma imagem contém computador, notebook ou máquina de escrever;
- classificar uma transação como fraude ou não fraude.

Prever tempo, preço, temperatura ou quantidade normalmente é **regressão**, pois a saída é numérica. Há uma nuance: prever a **probabilidade** de falha costuma fazer parte de um classificador probabilístico, embora a probabilidade em si seja contínua; ela representa a confiança/estimativa de pertencimento à classe.

## Binária e multiclasse

- **Binária:** duas classes, como `sim/não`.
- **Multiclasse:** três ou mais classes, como `gato/cão/cavalo`.

Algoritmos de classificação não se limitam à classificação binária. KNN e árvores de decisão suportam naturalmente múltiplas classes.

## Supervisão e predição

KNN e árvores de decisão, quando usados para classificação ou regressão, são:

- **supervisionados**, porque aprendem a partir de exemplos com alvo conhecido;
- **preditivos**, porque estimam a saída para novos registros.

---

# Parte II — K-Nearest Neighbors (KNN)

## 2.1 A ideia central

KNN prediz um novo exemplo usando os **K exemplos de treino mais próximos** segundo uma métrica de distância.

Na classificação:

1. calcula a distância do novo ponto aos exemplos de treino;
2. seleciona os K mais próximos;
3. agrega suas classes, normalmente por votação;
4. atribui a classe vencedora.

Na regressão, agrega valores numéricos, normalmente pela média ou média ponderada.

Portanto, KNN pode ser usado para **classificação e regressão**, mas não é, em sua forma usual, um algoritmo de agrupamento.

## 2.2 Instância, parametrização e treinamento

KNN é um modelo:

- **baseado em instâncias**: mantém os exemplos de treino e consulta-os na predição;
- **lazy learner**: quase não constrói um modelo durante o ajuste; o trabalho pesado ocorre na consulta;
- **não paramétrico**: não assume uma forma funcional fixa nem resume a solução em um número fixo de parâmetros aprendidos.

“Não paramétrico” não significa “sem hiperparâmetros”. KNN possui `n_neighbors`, `metric`, `p`, `weights` e outros hiperparâmetros.

Em `scikit-learn`, `fit` ainda é necessário: ele valida e armazena os dados e pode construir uma estrutura de busca. Contudo, não há aprendizagem de coeficientes como em regressão linear.

## 2.3 O parâmetro K

K é a **quantidade de vizinhos considerada na predição**.

### K pequeno

- fronteira de decisão mais irregular;
- baixa tendência de suavização;
- maior sensibilidade a ruído e outliers;
- menor viés e maior variância;
- risco de **overfitting**.

Com `K = 1`, cada ponto de treino tende a classificar a si próprio corretamente, mas isso não garante boa generalização.

### K grande

- fronteira mais suave;
- menor sensibilidade a pequenas variações;
- maior viés e menor variância;
- risco de **underfitting**;
- maior influência da classe majoritária.

Se `K = N`, todos os exemplos participam da votação e o classificador tende a sempre escolher a classe globalmente majoritária. K não deve exceder o número de amostras disponíveis no ajuste.

### Como escolher K

Escolha K usando validação cruzada, avaliando uma métrica adequada ao problema. Não existe um K universal. Em classificação binária, valores ímpares podem reduzir empates, mas não eliminam todos os tipos de empate e não substituem validação.

## 2.4 Pesos na votação

Em `KNeighborsClassifier`:

- `weights='uniform'`: cada vizinho tem um voto de mesmo peso;
- `weights='distance'`: vizinhos mais próximos recebem mais peso, usualmente de forma inversa à distância;
- uma função personalizada também pode gerar os pesos.

Peso por distância pode reduzir a influência de vizinhos distantes, mas pode aumentar a sensibilidade a ruído muito próximo e a pontos duplicados.

## 2.5 Empates

Empate pode ocorrer quando classes diferentes obtêm a mesma votação ou pontuação. As formas de lidar incluem:

- usar K ímpar em problemas binários;
- votar por distância;
- definir uma regra determinística de desempate;
- considerar probabilidades/frequências por classe;
- verificar o comportamento específico da biblioteca.

Em `scikit-learn`, não se deve inventar um hiperparâmetro `priority`: ele não existe em `KNeighborsClassifier`. Empates podem depender da ordenação interna das classes e dos vizinhos; por isso, a política precisa ser testada quando for relevante.

## 2.6 Métricas de distância

### Minkowski

\[
d(x,y)=\left(\sum_{i=1}^{m}|x_i-y_i|^p\right)^{1/p}
\]

- `p = 1`: Manhattan;
- `p = 2`: Euclidiana;
- no limite `p → ∞`: Chebyshev.

### Euclidiana

\[
d(x,y)=\sqrt{\sum_i(x_i-y_i)^2}
\]

É a distância em linha reta e penaliza diferenças grandes mais intensamente por causa do quadrado.

### Manhattan

\[
d(x,y)=\sum_i|x_i-y_i|
\]

É a soma dos deslocamentos por dimensão e pode ser mais robusta que a euclidiana diante de diferenças extremas.

### Chebyshev

\[
d(x,y)=\max_i|x_i-y_i|
\]

Considera apenas a maior diferença entre dimensões.

### Hamming

Conta ou mede a proporção de posições diferentes; é adequada a atributos binários ou categóricos devidamente representados.

### Mahalanobis

Leva em conta escalas e correlações por meio da matriz de covariância. Pode ser útil em dados correlacionados, mas exige estimativa estável/invertível da covariância e tem custo maior.

## 2.7 Escala dos atributos

Como KNN depende de distância, atributos com valores numericamente maiores podem dominar o cálculo. Exemplo: renda entre 0 e 100.000 pode eclipsar idade entre 18 e 80.

Escalonar não é uma obrigação matemática em todos os conjuntos, mas é uma necessidade prática quando escalas/unidades são diferentes.

Transformadores comuns:

- `StandardScaler`: média 0 e desvio-padrão 1; sensível a outliers;
- `MinMaxScaler`: mapeia para um intervalo, normalmente `[0,1]`; sensível aos extremos;
- `RobustScaler`: usa mediana e intervalo interquartil; mais robusto a outliers;
- `MaxAbsScaler`: divide pelo maior valor absoluto; útil quando se deseja preservar esparsidade.

O scaler deve ser ajustado **somente no conjunto de treino**, idealmente dentro de um `Pipeline`, para evitar vazamento de dados.

## 2.8 Outliers

Um outlier é uma observação incomum em relação ao padrão relevante dos dados. A definição depende do contexto, da distribuição e do objetivo; não é correto eliminar automaticamente qualquer ponto distante.

Outliers podem mudar a vizinhança e a votação, sobretudo com K pequeno. Estratégias:

- investigar se é erro, evento raro legítimo ou mudança de regime;
- corrigir erros quando houver evidência;
- usar escalonamento robusto;
- considerar distância/pesos mais robustos;
- remover somente com justificativa documentada;
- validar o efeito da decisão.

## 2.9 Desbalanceamento de classes

Quando uma classe é muito mais frequente, a votação tende a favorecê-la. Formas de mitigação:

- reamostragem do treino;
- votação ponderada por distância e/ou classe;
- escolha criteriosa de K;
- métricas como recall, precisão, F1, PR-AUC e balanced accuracy;
- ajuste de decisão conforme o custo do erro.

Aplicar “logaritmo nas variáveis” não é uma solução geral para desbalanceamento; isso altera distribuições de atributos, não a proporção das classes.

## 2.10 Custo computacional

No brute force, uma consulta compara o novo ponto com todas as `N` amostras em `D` dimensões:

\[
O(ND)
\]

Se D for tratado como constante, costuma-se dizer simplesmente **linear em N**. Para várias consultas, o custo se multiplica pelo número de consultas; localizar/ordenar vizinhos também adiciona custo.

KNN costuma ter:

- ajuste barato;
- memória alta, pois mantém o treino;
- predição cara.

### Estruturas de busca

- `brute`: cálculo direto; pode ser competitivo em alta dimensão ou com poucos dados;
- `KDTree`: particiona recursivamente o espaço por eixos; costuma ser útil em baixa dimensão;
- `BallTree`: organiza pontos em regiões/“bolas”; pode funcionar melhor com certas métricas e dimensões moderadas;
- métodos aproximados como HNSW são alternativas para grandes bases, embora não sejam o backend padrão do KNN clássico do `scikit-learn`.

Árvores perdem eficiência em dimensão alta por causa da **maldição da dimensionalidade**: distâncias tornam-se menos discriminativas e a poda da busca enfraquece.

Parâmetros como `leaf_size` influenciam tempo de construção, consulta e memória. Não existe “quanto menor, sempre melhor”.

## 2.11 Pontos fortes e limitações

Pontos fortes:

- simples e intuitivo;
- não impõe fronteira funcional fixa;
- naturalmente multiclasse;
- útil como baseline em bases pequenas/médias.

Limitações:

- consulta e memória caras;
- sensível a escala, métrica, K, ruído e atributos irrelevantes;
- piora em alta dimensão;
- explicação local é possível, mas o modelo global não é resumido por uma fórmula compacta.

---

# Parte III — Árvores de Decisão

## 3.1 A ideia central

Uma árvore aprende uma sequência hierárquica de regras condicionais. Cada divisão busca produzir filhos mais puros em relação ao alvo.

Estrutura:

- **raiz:** primeiro teste;
- **nó de decisão/interno:** novo teste sobre um atributo;
- **ramo:** resultado de um teste;
- **folha:** predição final.

Cada caminho da raiz a uma folha forma uma regra `SE ... ENTÃO ...`. A estrutura é um grafo direcionado acíclico com organização de árvore; cada nó, exceto a raiz, possui um único pai.

## 3.2 Natureza do algoritmo

Árvores de decisão são:

- supervisionadas;
- preditivas;
- não paramétricas;
- de aprendizado **eager**, pois constroem a estrutura no ajuste;
- gulosas na construção usual: escolhem o melhor corte local em cada etapa.

“Guloso” significa que o algoritmo escolhe a melhor divisão imediata, sem garantir que a árvore final seja globalmente ótima.

Podem resolver:

- classificação binária;
- classificação multiclasse;
- regressão.

Não se limitam a um tipo de tarefa nem a dados exclusivamente numéricos ou exclusivamente categóricos. A implementação pode exigir codificação para categorias; por exemplo, árvores do `scikit-learn` trabalham com entrada numérica.

## 3.3 Famílias de algoritmos

- **ID3:** tradicionalmente usa ganho de informação/entropia e atributos categóricos;
- **C4.5:** extensão do ID3, associada a gain ratio e tratamento mais amplo de atributos;
- **C5.0:** evolução comercial/otimizada da família C4.5;
- **CART:** produz divisões binárias; usa tipicamente Gini em classificação e erro quadrático em regressão.

As implementações concretas diferem; não se deve atribuir toda propriedade de uma família automaticamente a qualquer biblioteca.

## 3.4 Impureza e qualidade do corte

### Índice de Gini

\[
Gini(S)=1-\sum_{i=1}^{C}p_i^2
\]

### Entropia

\[
H(S)=-\sum_{i=1}^{C}p_i\log_2(p_i)
\]

Ambas valem 0 em um nó puro. Quanto maior a mistura de classes, maior a impureza.

O ganho de informação é a redução ponderada de entropia:

\[
IG = H(parent)-\sum_j\frac{|S_j|}{|S|}H(S_j)
\]

Para Gini, usa-se de modo análogo a redução ponderada de impureza. A melhor divisão maximiza a redução de impureza.

Em geral, Gini é um pouco mais barato porque evita logaritmos. Entropia e Gini frequentemente escolhem cortes semelhantes, mas não são idênticos.

Não confundir essas métricas com índice de Dunn, Davies–Bouldin ou silhouette, que são métricas comuns de agrupamento.

## 3.5 Exemplo mental para escolher a raiz

Para cada atributo/corte candidato:

1. calcule a impureza do nó pai;
2. separe os registros conforme o corte;
3. calcule a impureza de cada filho;
4. tire a média ponderada pelo tamanho dos filhos;
5. subtraia do valor do pai;
6. escolha o maior ganho/redução.

Na tabela clássica `Play Tennis`, quando se usa ganho de informação, `Outlook` costuma ser a melhor raiz. O importante é saber demonstrar o cálculo, não apenas memorizar o nome.

## 3.6 Escala e outliers

Árvores fazem cortes do tipo `x_j ≤ t`; uma transformação monotônica preserva a ordem dos valores e, portanto, normalmente preserva as divisões possíveis. Assim:

- padronização/normalização geralmente não é necessária;
- unidades diferentes não dominam uma “distância”, pois não há distância global;
- não se deve dizer que MinMaxScaler é obrigatório para limitar a quantidade de regiões.

Outliers ainda podem afetar os limiares, criar folhas pequenas e favorecer splits específicos. A árvore não é imune, mas costuma ser menos sensível à escala e a valores extremos que KNN ou modelos baseados em distância.

## 3.7 Viés, variância e profundidade

Uma árvore profunda e sem restrições tende a ter:

- baixo viés no treino;
- alta variância;
- risco de overfitting;
- regras muito específicas e folhas com poucas amostras.

Uma árvore rasa tende a ter maior viés e menor variância, podendo sofrer underfitting.

`max_depth` é o comprimento máximo permitido do caminho da raiz até uma folha, medido em níveis/divisões conforme a convenção da implementação. Não é a quantidade total de nós ou folhas.

## 3.8 Critérios de parada e regularização

Possíveis critérios:

- atingir `max_depth`;
- nó ter menos amostras que o necessário para dividir (`min_samples_split`);
- folha precisar manter um mínimo (`min_samples_leaf`);
- ganho/queda de impureza ser insuficiente (`min_impurity_decrease`);
- nó já estar puro;
- não existir divisão válida;
- limite de folhas (`max_leaf_nodes`).

## 3.9 Como combater overfitting

### Pré-poda

Restringe o crescimento durante o treino:

- `max_depth`;
- `min_samples_split`;
- `min_samples_leaf`;
- `max_leaf_nodes`;
- `min_impurity_decrease`;
- `max_features`.

### Pós-poda

Constrói a árvore e remove ramos pouco úteis. Em CART/scikit-learn, `ccp_alpha` controla a poda por complexidade de custo.

### Ensembles

- Random Forest reduz variância combinando árvores treinadas com amostras e subconjuntos de atributos;
- boosting combina árvores sequencialmente, corrigindo erros anteriores.

Ensembles costumam melhorar generalização, mas diminuem a interpretabilidade direta de uma única árvore.

## 3.10 Desbalanceamento

Árvores também sofrem com classes desbalanceadas: a função de impureza e as folhas podem favorecer a classe majoritária. Possíveis respostas:

- `class_weight`/pesos de amostra;
- reamostragem dentro apenas do treino;
- critérios e limiares alinhados ao custo do erro;
- métricas apropriadas;
- folhas mínimas e validação cruzada estratificada.

## 3.11 Complexidade

Para uma implementação eficiente, construir uma árvore balanceada é frequentemente descrito, de forma aproximada, como `O(N log N)` por atributo ou `O(D N log N)`, dependendo de ordenação, algoritmo e convenções. O pior caso pode se aproximar de `O(D N²)` em árvore extremamente desbalanceada/implementação ingênua.

A predição custa aproximadamente:

- `O(profundidade)` por exemplo;
- `O(log N)` se a árvore for balanceada;
- `O(N)` no pior caso degenerado.

Portanto, a resposta “linear, quadrático ou logarítmico” só é correta quando a pergunta deixa claro se fala de treino, predição, caso médio ou pior caso.

## 3.12 Pontos fortes e limitações

Pontos fortes:

- regras interpretáveis;
- não exige escalonamento na maioria dos casos;
- modela relações não lineares e interações;
- serve para classificação e regressão;
- predição rápida.

Limitações:

- alta variância e instabilidade: pequenas mudanças podem gerar outra estrutura;
- overfitting se crescer sem controle;
- construção gulosa não garante ótimo global;
- fronteiras alinhadas aos eixos podem exigir muitos cortes;
- categorias precisam de tratamento compatível com a implementação.

---

# Parte IV — Comparação que deve sair de memória

| Aspecto | KNN | Árvore de decisão |
|---|---|---|
| Aprendizagem | Lazy, baseada em instâncias | Eager, constrói regras |
| Parametrização | Não paramétrico | Não paramétrico |
| Treino | Barato, salvo estrutura de busca | Mais caro, busca cortes |
| Predição | Potencialmente cara | Geralmente rápida |
| Escalonamento | Muito importante | Geralmente desnecessário |
| Outliers | Pode ser muito sensível | Pode afetar splits, mas tende a ser menos sensível |
| Alta dimensão | Sofre fortemente | Pode selecionar atributos, mas ainda pode overfit |
| Interpretação | Local: vizinhos | Global/local: regras e caminhos |
| Overfitting | K pequeno | Árvore profunda |
| Underfitting | K grande | Árvore rasa/restrita |
| Desbalanceamento | Votação favorece maioria | Impureza/folhas favorecem maioria |
| Classificação/regressão | Ambas | Ambas |

## Comparação aproximada de treinamento

- KNN brute force: ajuste próximo de `O(ND)`, pois principalmente armazena/valida os dados;
- Naive Bayes: treinamento aproximadamente `O(ND)`, estimando estatísticas por classe;
- árvore: frequentemente perto de `O(DN log N)` em cenários eficientes, com pior caso superior.

Essas expressões dependem da variante e da implementação. A ordem normalmente esperada em uma questão simplificada de custo de treino é **KNN/Naive Bayes antes da árvore**, mas KNN transfere grande parte do custo para a predição.

---

# Parte V — Respostas-modelo para falar com segurança

## “Por que KNN precisa de escala?”

> Porque sua decisão depende diretamente de distâncias. Se um atributo possui magnitude muito maior, ele pode dominar a medida mesmo sem ser mais importante. Eu ajustaria o scaler apenas no treino, dentro de um pipeline, e escolheria scaler, métrica e K por validação cruzada. A exceção é quando os atributos já estão em escalas comparáveis ou quando a métrica foi deliberadamente construída para incorporar as unidades.

## “KNN é paramétrico?”

> Não. Ele não supõe uma forma funcional fixa nem aprende um vetor de parâmetros de tamanho fixo; armazena instâncias e decide localmente. Contudo, possui hiperparâmetros, como K, métrica, p e pesos. Não paramétrico não significa sem configuração.

## “O que acontece quando K aumenta?”

> A fronteira fica mais suave: a variância cai e o viés sobe. Isso reduz a sensibilidade a ruído, mas K excessivo causa underfitting e favorece a classe majoritária. Eu escolheria K por validação, não por uma regra universal.

## “Como uma árvore escolhe um corte?”

> Ela avalia atributos e limiares candidatos e escolhe o que mais reduz a impureza ponderada dos filhos. Em classificação, critérios comuns são Gini e entropia. É uma escolha gulosa: ótima naquele nó, mas não necessariamente para a árvore inteira.

## “Árvore precisa de normalização?”

> Em geral, não. A árvore usa comparações e limiares por atributo, não distâncias; transformações monotônicas preservam a ordem dos valores. Ainda assim, pré-processamento pode ser necessário para dados ausentes, categorias ou exigências específicas da implementação.

## “Como controlar overfitting em árvores?”

> Posso limitar profundidade, tamanho mínimo dos nós/folhas, número de folhas e ganho mínimo; também posso aplicar pós-poda com `ccp_alpha`. Seleciono os valores por validação cruzada e comparo treino versus validação. Para reduzir variância de forma mais forte, considero ensembles como Random Forest.

---

# Parte VI — Armadilhas presentes nas questões

1. **Probabilidade é contínua, mas pode ser saída de um classificador.** Não confundir natureza do score com natureza do alvo.
2. **Não paramétrico não significa sem hiperparâmetros.**
3. **`fit` de KNN não é “não fazer nada”.** Há validação, armazenamento e, possivelmente, indexação.
4. **K ímpar não elimina todo empate.** Ajuda sobretudo em classificação binária com voto uniforme.
5. **KNN não exige normalização por definição matemática**, mas quase sempre exige atenção à escala na prática.
6. **Outlier não deve ser apagado automaticamente.** Primeiro se investiga sua natureza.
7. **KDTree não é sempre melhor que brute force.** Dimensão, métrica, N e hardware mudam a escolha.
8. **Árvore não é somente binária nem somente classificadora.** CART usa splits binários, mas a tarefa pode ser multiclasse ou regressão.
9. **Árvore não é imune a outliers ou desbalanceamento.** Apenas reage de forma diferente de modelos de distância.
10. **Gini e entropia medem impureza; Dunn, silhouette e Davies–Bouldin avaliam clustering.**
11. **Complexidade precisa de contexto:** treino/predição, médio/pior caso, N/D e implementação.
12. **Menor erro de treino não significa melhor modelo.** O objetivo é generalização.

---

# Parte VII — Roteiro de treinamento

## Sessão 1 — Fundamentos e enquadramento

- diferenciar classificação, regressão e clustering;
- reconhecer alvo categórico e score probabilístico;
- explicar supervisão, binário e multiclasse.

Critério de domínio: classificar corretamente dez problemas novos e justificar cada escolha.

## Sessão 2 — KNN de ponta a ponta

- simular uma predição manual;
- variar K e explicar viés/variância;
- comparar métricas;
- explicar escala, outliers, empate e desbalanceamento;
- discutir custo e estruturas de busca.

Critério de domínio: responder cada tema com mecanismo, consequência e exceção.

## Sessão 3 — Árvores de ponta a ponta

- desenhar raiz, nós, ramos e folhas;
- calcular Gini e entropia;
- escolher um corte por ganho;
- diagnosticar profundidade e poda;
- discutir estabilidade, desbalanceamento e ensembles.

Critério de domínio: construir manualmente os primeiros níveis de uma árvore pequena.

## Sessão 4 — Comparação e seleção de modelo

- escolher KNN ou árvore para diferentes cenários;
- justificar usando N, D, latência, interpretabilidade e tipo de variável;
- propor pipeline e validação sem vazamento.

Critério de domínio: defender a escolha e também explicar quando ela deixaria de ser adequada.

## Sessão 5 — Simulado oral

Para cada resposta, usar a estrutura:

1. resposta direta em uma frase;
2. mecanismo técnico;
3. efeito prático;
4. exemplo;
5. ressalva ou exceção;
6. como validar na prática.

---

# Checklist final de domínio

Você domina o conteúdo quando consegue, sem consultar material:

- explicar por que KNN é supervisionado, preditivo, não paramétrico, lazy e baseado em instâncias;
- calcular Euclidiana, Manhattan, Minkowski e interpretar Hamming/Mahalanobis;
- prever o efeito de K, pesos, escala, outliers e desbalanceamento;
- explicar brute force, KDTree, BallTree e maldição da dimensionalidade;
- descrever uma árvore como regras hierárquicas e explicar sua natureza gulosa;
- calcular Gini, entropia e ganho ponderado;
- distinguir ID3, C4.5/C5.0 e CART em nível conceitual;
- relacionar profundidade a viés, variância, overfitting e underfitting;
- listar e justificar pré-poda, pós-poda e ensembles;
- comparar KNN e árvore levando em conta treino, predição, memória, interpretação e dados;
- detectar alternativas enganosas e explicar por que estão erradas.
