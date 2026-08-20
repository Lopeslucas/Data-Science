# Guia de domínio — Classificação e K-Nearest Neighbors (KNN)

Material consolidado a partir das perguntas da lista sobre **classificação, KNN, métricas de distância, escala, outliers, desbalanceamento e estruturas de busca**.

O objetivo não é apenas identificar a alternativa correta. Uma resposta segura deve seguir:

> **definição → mecanismo → consequência → exemplo → limitação → validação prática**

---

# Parte I — Fundamentos de classificação

## 1.1 O que é classificação?

Classificação é uma tarefa supervisionada em que a variável-alvo é **categórica**. O modelo aprende com exemplos rotulados e prediz a classe de novas observações.

As entradas podem ser:

- numéricas;
- categóricas, após representação compatível;
- uma combinação das duas.

Exemplos:

- prever `falha` ou `não falha` de um computador;
- identificar `computador`, `notebook` ou `máquina de escrever` em uma imagem;
- classificar uma transação como `fraude` ou `não fraude`.

Prever tempo, quantidade ou preço normalmente é regressão. Prever a **probabilidade de falha** pode fazer parte de um classificador probabilístico: a probabilidade é contínua, mas representa a confiança de pertencimento a uma classe categórica.

## 1.2 Binária e multiclasse

- **Classificação binária:** duas classes.
- **Classificação multiclasse:** três ou mais classes.

É falso afirmar que classificadores resolvem somente problemas binários. KNN suporta naturalmente múltiplas classes.

## 1.3 Classificação, regressão e agrupamento

- **Classificação:** prediz uma categoria conhecida no treino.
- **Regressão:** prediz um valor numérico.
- **Agrupamento:** descobre grupos sem um alvo rotulado.

KNN, em sua forma usual, serve para classificação e regressão. Não deve ser confundido com K-Means: ambos usam a letra K, mas resolvem problemas diferentes.

---

# Parte II — Como o KNN funciona

## 2.1 Ideia central

KNN significa **K-Nearest Neighbors**, ou K vizinhos mais próximos. Ele parte da hipótese de que observações próximas em um espaço de atributos tendem a possuir respostas semelhantes.

Para classificar uma nova observação:

1. represente-a no mesmo espaço dos dados de treino;
2. calcule sua distância aos exemplos de treino;
3. selecione os `K` menores valores;
4. agregue as classes desses vizinhos;
5. escolha a classe vencedora.

Na regressão, substitui-se a votação por média ou média ponderada dos valores dos vizinhos.

## 2.2 Natureza do algoritmo

KNN é:

- **supervisionado**, porque utiliza alvos conhecidos;
- **preditivo**, porque estima respostas de novos registros;
- **não paramétrico**, pois não assume uma função de forma fixa;
- **baseado em instâncias**, pois retém exemplos do treino;
- **lazy learner**, porque transfere grande parte do trabalho para a predição;
- capaz de realizar **classificação e regressão**.

Logo, marcar “KNN não é preditivo” é incorreto.

## 2.3 O que acontece em `fit`?

Dizer que KNN “não treina” é um atalho. Em bibliotecas como `scikit-learn`, `fit`:

- valida os dados;
- armazena atributos e alvos;
- registra classes e metadados;
- pode construir KDTree ou BallTree.

Ele não aprende coeficientes como uma regressão linear, mas ainda possui uma fase de ajuste.

## 2.4 Premissa de localidade

A premissa central é que a distância escolhida representa **similaridade relevante para o alvo**. Isso exige:

- atributos informativos;
- representação coerente das variáveis;
- escala adequada;
- uma métrica compatível com os dados;
- densidade local suficiente;
- estabilidade razoável entre treino e dados futuros.

Se “próximo” geometricamente não significa “semelhante” para o problema, KNN não funcionará bem.

---

# Parte III — O hiperparâmetro K

## 3.1 O que K significa?

K é o número de vizinhos usados na predição. Ele não representa número de grupos, iterações, distâncias calculadas ou pontos removidos.

## 3.2 K pequeno

Quando `K` se aproxima de 1:

- a fronteira de decisão fica irregular;
- o erro de treino tende a cair;
- o modelo fica mais sensível a ruído e outliers locais;
- o viés tende a diminuir;
- a variância tende a aumentar;
- cresce o risco de overfitting.

Com `K=1`, cada observação de treino geralmente é seu próprio vizinho e tende a ser classificada corretamente. Isso não garante generalização.

## 3.3 K grande

Quando `K` cresce:

- a fronteira fica mais suave;
- a variância diminui;
- o viés aumenta;
- detalhes locais são apagados;
- cresce o risco de underfitting;
- a classe majoritária ganha influência.

Se `K=N` e a votação é uniforme, todos os exemplos de treino votam. O modelo tende a predizer sempre a classe globalmente majoritária. No exemplo das imagens, com frequências `A=20%`, `B=15%`, `C=35%` e `D=30%`, a resposta seria `C`.

Com peso por distância, `K=N` não implica necessariamente uma previsão constante, pois pontos próximos ainda recebem mais peso.

## 3.4 Como escolher K

Não existe um K universal. Selecione-o por validação cruzada, junto com:

- métrica de distância;
- esquema de pesos;
- scaler;
- atributos usados;
- métrica de avaliação.

Valores ímpares ajudam a reduzir empates em classificação binária com voto uniforme, mas não eliminam todos os empates e não substituem validação.

---

# Parte IV — Votação e pesos

## 4.1 `weights='uniform'`

Todos os vizinhos têm o mesmo peso. A classe mais frequente entre os K vizinhos vence.

## 4.2 `weights='distance'`

Vizinhos mais próximos exercem maior influência, normalmente com peso inversamente proporcional à distância.

Esse esquema pode:

- preservar melhor estruturas locais;
- reduzir a influência de vizinhos distantes;
- ajudar em densidades não uniformes;
- aumentar a influência de ruído ou rótulo incorreto extremamente próximo.

`weights='distance'` não altera K automaticamente. Os K vizinhos continuam sendo selecionados; apenas seus votos recebem pesos diferentes.

## 4.3 Função personalizada

No `scikit-learn`, `weights` também pode receber uma função que transforma distâncias em pesos. Ela deve ser validada para evitar instabilidade com distância zero.

## 4.4 Empates

Há dois tipos diferentes de empate:

- vizinhos diferentes à mesma distância no limite de seleção;
- classes com a mesma soma de votos ou pesos.

Possíveis tratamentos:

- K ímpar em caso binário;
- voto ponderado por distância;
- regra de desempate explicitamente definida;
- escolha determinística da biblioteca;
- análise de probabilidades/frequências por classe.

Não existe um parâmetro `priority` em `KNeighborsClassifier`. O classificador não cria uma classe `tie`. Na implementação do `scikit-learn`, empates de pontuação entre classes são resolvidos de forma determinística pela ordenação interna das classes, não simplesmente pela ordem das linhas do treino.

## 4.5 Probabilidades do KNN

Na classificação, `predict_proba` deriva da proporção ou soma ponderada das classes na vizinhança. Isso fornece uma estimativa local, mas não garante boa calibração probabilística.

Com K pequeno, as probabilidades assumem poucos valores discretos e podem ser instáveis.

---

# Parte V — Métricas de distância

## 5.1 Minkowski

\[
d(x,y)=\left(\sum_{i=1}^{D}|x_i-y_i|^p\right)^{1/p}
\]

- `p=1`: Manhattan;
- `p=2`: Euclidiana;
- no limite `p→∞`: Chebyshev.

Chebyshev não corresponde a `p=1` nem a `p=2`.

## 5.2 Euclidiana

\[
d(x,y)=\sqrt{\sum_{i=1}^{D}(x_i-y_i)^2}
\]

É a distância em linha reta. O quadrado dá influência maior a diferenças grandes.

Para `A=(1,2)` e `B=(3,4)`:

\[
d(A,B)=\sqrt{(1-3)^2+(2-4)^2}=\sqrt{8}\approx2{,}828
\]

## 5.3 Manhattan

\[
d(x,y)=\sum_{i=1}^{D}|x_i-y_i|
\]

Para os mesmos pontos:

\[
d(A,B)=|1-3|+|2-4|=4
\]

Pode ser menos sensível que a Euclidiana a diferenças extremas, mas não é automaticamente superior em alta dimensão.

## 5.4 Chebyshev

\[
d(x,y)=\max_i|x_i-y_i|
\]

Considera apenas a maior diferença entre dimensões.

## 5.5 Hamming

Conta ou mede a proporção de posições diferentes. É apropriada para vetores binários e algumas representações categóricas.

Usar Hamming em códigos categóricos exige cuidado: a representação deve preservar o significado desejado e não inventar uma ordem.

## 5.6 Mahalanobis

\[
d(x,y)=\sqrt{(x-y)^T\Sigma^{-1}(x-y)}
\]

Considera escalas e correlações por meio da matriz de covariância. Pode ser útil quando atributos são correlacionados, mas:

- exige uma covariância estimável e invertível ou regularizada;
- é mais custosa que Euclidiana;
- fica instável com muitas dimensões e poucos dados;
- a covariância deve ser estimada apenas no treino.

## 5.7 Como escolher a métrica

Escolha com base em:

- tipo das variáveis;
- significado de uma unidade de diferença;
- presença de correlações;
- esparsidade;
- dimensionalidade;
- robustez desejada;
- custo computacional;
- validação cruzada.

Misturar variáveis numéricas e categóricas pode exigir uma métrica específica, como Gower, ou engenharia de representação. One-hot encoding com Euclidiana nem sempre produz a semântica ideal.

---

# Parte VI — Escala e pré-processamento

## 6.1 Por que escala importa?

Como KNN depende de distância, um atributo com grande magnitude pode dominar o cálculo mesmo sem ser mais relevante.

Exemplo:

- idade: `18–80`;
- renda: `0–100.000`.

Sem escala, a renda provavelmente dominará a distância Euclidiana.

## 6.2 É obrigatório escalonar?

Não é uma obrigação matemática universal. É uma necessidade prática quando:

- as unidades são diferentes;
- as magnitudes são muito distintas;
- a métrica não corrige a escala;
- não existe justificativa de negócio para pesos implícitos.

Se todos os atributos já estão em escalas comparáveis ou a métrica foi desenhada deliberadamente, o escalonamento pode ser desnecessário.

## 6.3 Principais transformadores

### `StandardScaler`

\[
z=\frac{x-\mu}{\sigma}
\]

Produz média próxima de 0 e variância próxima de 1 no treino. É incorreto afirmar “média 1 e variância 0”. É sensível a outliers.

### `MinMaxScaler`

Mapeia o intervalo observado para `[0,1]` ou outro intervalo definido. Outliers comprimem a maior parte dos dados, mas não “binarizam” os valores.

### `RobustScaler`

Usa mediana e intervalo interquartil. É mais robusto a valores extremos marginais.

### `MaxAbsScaler`

Divide pelo maior valor absoluto e pode preservar esparsidade.

## 6.4 Vazamento de dados

O scaler deve ser ajustado somente com o treino. Em validação cruzada, ele deve ser reajustado dentro de cada fold.

Use um `Pipeline`:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])
```

Ajustar o scaler antes de separar dados permite que validação/teste influenciem o treino.

## 6.5 Escala não seleciona atributos

Colocar variáveis na mesma escala evita dominância causada apenas por unidades, mas não impede que atributos irrelevantes prejudiquem a vizinhança. Seleção de atributos, redução de dimensionalidade e conhecimento de domínio continuam importantes.

---

# Parte VII — Outliers e ruído

## 7.1 O que é outlier?

Um outlier é uma observação incomum em relação ao padrão relevante. A definição depende do contexto, distribuição e objetivo.

A frase “outlier é uma observação que **não** está a uma distância anormal” está invertida. Ainda assim, distância isolada não basta: um ponto raro pode ser erro ou evento legítimo.

## 7.2 Como outliers afetam KNN

- podem se tornar vizinhos enganosos;
- distorcem scalers baseados em média, desvio, mínimo ou máximo;
- afetam especialmente `K` pequeno;
- na regressão, um alvo extremo pode alterar fortemente a média local;
- um outlier isolado e distante pode ter pouco impacto em regiões densas.

## 7.3 O que fazer

1. investigar origem e contexto;
2. verificar erro de medição, digitação ou integração;
3. comparar distribuição e grupos;
4. corrigir quando houver evidência;
5. remover apenas com justificativa documentada;
6. considerar `RobustScaler`, métrica adequada ou transformação;
7. validar o impacto da decisão.

Colocar outliers propositalmente apenas no teste é vazamento/manipulação da avaliação. Trocar a semente ou usar KDTree não resolve sua natureza.

Peso por distância reduz a influência de outliers distantes, mas aumenta a de um outlier muito próximo. Usar `K≈1` geralmente piora a sensibilidade.

---

# Parte VIII — Desbalanceamento de classes

## 8.1 O problema

Uma classe é minoritária quando possui bem menos exemplos que outra. No KNN, a classe majoritária tende a aparecer com mais frequência nas vizinhanças, principalmente com K grande.

## 8.2 Por que acurácia pode enganar

Se 99% dos casos não são fraude, predizer sempre “não fraude” produz 99% de acurácia e zero utilidade para encontrar fraudes.

Use métricas alinhadas ao custo:

- recall/sensibilidade;
- precisão;
- F1;
- PR-AUC;
- balanced accuracy;
- matriz de confusão;
- custo esperado dos erros.

## 8.3 Estratégias

- sobreamostragem ou subamostragem apenas dentro do treino;
- vizinhança ponderada por distância;
- pesos adicionais por classe via função personalizada;
- escolha criteriosa de K;
- ajuste do limiar sobre probabilidades locais;
- validação cruzada estratificada;
- variantes especializadas de nearest neighbors.

Peso por distância pode ajudar, mas não é solução geral para desbalanceamento. Aplicar logaritmo nas variáveis altera atributos, não a frequência das classes.

---

# Parte IX — Custo computacional

## 9.1 Notação

- `N`: amostras de treino;
- `D`: dimensões/atributos;
- `Q`: consultas;
- `K`: vizinhos retornados.

## 9.2 Brute force

Uma consulta calcula a distância para todas as amostras:

\[
O(ND)
\]

Se `D` é tratado como constante, diz-se aproximadamente linear em `N`. Para `Q` consultas:

\[
O(QND)
\]

Encontrar os K menores acrescenta custo de seleção/ordenação, dependendo da implementação.

## 9.3 Treino, predição e memória

- ajuste simples: aproximadamente `O(ND)` para validar/copiar dados;
- construção de índice: custo adicional;
- predição brute force: `O(ND)` por consulta;
- memória: `O(ND)` para manter o treino.

Logo, KNN tem treino relativamente barato, predição cara e uso elevado de memória.

## 9.4 Por que “o custo é linear” é incompleto

A resposta depende de:

- treino ou predição;
- número de consultas;
- dimensionalidade;
- busca exata ou aproximada;
- estrutura de índice;
- custo da métrica;
- seleção dos K vizinhos.

---

# Parte X — Estruturas de busca

## 10.1 Brute force

Compara diretamente a consulta a todos os pontos. Pode ser a melhor escolha quando:

- N é pequeno;
- D é alto;
- dados são densos e vetorização é eficiente;
- a métrica não é suportada pelas árvores;
- o overhead de indexação não compensa.

## 10.2 KDTree

KDTree é uma árvore binária que particiona recursivamente o espaço com cortes alinhados aos eixos.

Na consulta, usa limites geométricos e desigualdade triangular para eliminar regiões que não podem conter vizinhos melhores. Não precisa calcular a distância completa para todos os pontos.

Funciona melhor em baixa dimensionalidade. Em alta dimensão, a poda perde eficiência e a busca se aproxima de brute force.

## 10.3 BallTree

BallTree organiza pontos em regiões delimitadas por centros e raios, isto é, “bolas” no espaço métrico. Pode suportar métricas e geometrias para as quais KDTree é inadequada e pode funcionar melhor em dimensões moderadas.

É incorreto dizer que BallTree foi criada para corrigir a ineficiência da KDTree **em baixa dimensão**. KDTree já costuma funcionar bem em baixa dimensão; BallTree pode ser vantajosa em outras geometrias, métricas e dimensões moderadas.

## 10.4 `leaf_size`

Controla aproximadamente quantos pontos ficam nas folhas.

- folha muito pequena: árvore maior, construção/memória maiores e mais travessia;
- folha muito grande: menos poda refinada e mais busca brute force dentro da folha;
- valor próximo de N: comportamento mais parecido com brute force.

Não existe “quanto menor, sempre mais rápido”. O valor ideal depende dos dados, hardware e padrão de consultas.

## 10.5 Complexidade das árvores

Em condições favoráveis e baixa dimensão, a consulta pode se aproximar de comportamento logarítmico, mas não há garantia universal. O pior caso pode exigir visitar grande parte dos dados.

## 10.6 Busca aproximada

Para bases grandes, métodos como HNSW, IVF ou bibliotecas especializadas trocam exatidão por latência e memória. Eles não são os backends clássicos padrão de `KNeighborsClassifier`, mas são relevantes em aplicações reais.

## 10.7 Maldição da dimensionalidade

À medida que D cresce:

- o volume do espaço aumenta rapidamente;
- os dados ficam esparsos;
- distâncias do vizinho mais próximo e distante se tornam parecidas;
- é necessário muito mais dado para representar regiões locais;
- KDTree/BallTree podam menos;
- atributos irrelevantes acumulam ruído.

Possíveis respostas:

- seleção de atributos;
- PCA ou outra redução de dimensionalidade ajustada no treino;
- métrica apropriada;
- regularização da representação;
- mais dados;
- trocar de modelo quando a proximidade deixa de ser informativa.

---

# Parte XI — KNeighborsClassifier versus RadiusNeighborsClassifier

## 11.1 KNeighborsClassifier

Usa exatamente K vizinhos, independentemente da densidade local.

É adequado quando a amostragem é relativamente uniforme e sempre se deseja um número fixo de votos.

## 11.2 RadiusNeighborsClassifier

Usa todos os vizinhos dentro de um raio fixo. A quantidade varia por consulta.

Pode ser útil quando a densidade é irregular, mas apresenta riscos:

- regiões densas têm muitos vizinhos;
- regiões esparsas podem não ter nenhum;
- o raio é sensível à escala e dimensionalidade;
- ainda pode sofrer com outliers.

Ele não recebe K como hiperparâmetro principal; recebe `radius`.

## 11.3 Escolha entre os dois

- K fixo: garante vizinhos, mas pode buscar pontos muito distantes em regiões esparsas.
- raio fixo: preserva uma noção absoluta de proximidade, mas pode não encontrar ninguém.

Compare por validação e trate explicitamente consultas sem vizinhos.

---

# Parte XII — Parâmetros versus hiperparâmetros

## 12.1 Parâmetros aprendidos

KNN não aprende coeficientes ou uma árvore de decisão. Seu estado ajustado contém:

- dados de treino armazenados;
- alvos/classes;
- metadados da representação;
- eventualmente uma estrutura de busca.

## 12.2 Hiperparâmetros principais

No `KNeighborsClassifier`:

- `n_neighbors`: K;
- `weights`: `uniform`, `distance` ou função;
- `metric`: distância;
- `p`: potência da Minkowski;
- `metric_params`: argumentos adicionais;
- `algorithm`: `auto`, `ball_tree`, `kd_tree` ou `brute`;
- `leaf_size`: tamanho das folhas;
- `n_jobs`: paralelismo operacional.

No `RadiusNeighborsClassifier`, `radius` substitui o número fixo de vizinhos como conceito central.

## 12.3 Não paramétrico não significa sem configuração

KNN é não paramétrico porque não assume uma forma funcional fixa e sua representação cresce com os dados. Ainda assim, possui hiperparâmetros e estado aprendido.

---

# Parte XIII — Pontos positivos e negativos

## 13.1 Pontos positivos

- simples e intuitivo;
- quase nenhum pressuposto sobre a forma da fronteira;
- classificação e regressão;
- naturalmente multiclasse;
- adapta-se a padrões locais não lineares;
- treino conceitualmente barato;
- explicação local pelos vizinhos;
- útil como baseline em bases pequenas e médias.

## 13.2 Pontos negativos e como lidar

- **Predição cara:** KDTree/BallTree em condições favoráveis, busca aproximada, redução de dados ou outro modelo.
- **Memória alta:** protótipos, condensação, amostragem ou modelo paramétrico.
- **Sensibilidade à escala:** scaler dentro de pipeline.
- **Sensibilidade a K e métrica:** validação cruzada conjunta.
- **Outliers e ruído:** diagnóstico, limpeza justificada, K/pesos/métrica robustos.
- **Alta dimensão:** seleção/redução de atributos ou troca de modelo.
- **Desbalanceamento:** reamostragem, pesos, limiar e métricas adequadas.
- **Atributos irrelevantes:** seleção de features e conhecimento de domínio.
- **Densidade variável:** peso por distância, radius neighbors ou métodos adaptativos.
- **Pouca extrapolação:** KNN prediz a partir de exemplos vistos; para extrapolar tendências, considere outro modelo.
- **Probabilidade pouco calibrada:** avaliação e calibração apropriada.

---

# Parte XIV — Premissas e quando usar

## 14.1 Premissas práticas

- proximidade deve refletir similaridade do alvo;
- há exemplos suficientes nas regiões de interesse;
- atributos são representados e escalados adequadamente;
- a dimensionalidade efetiva é controlada;
- treino e produção pertencem a regimes comparáveis;
- custo de consulta e memória são aceitáveis.

KNN não exige normalidade dos dados nem uma fronteira linear.

## 14.2 Quando usar

- base pequena ou média;
- dimensionalidade baixa/moderada;
- padrão local relevante;
- fronteira não linear;
- baseline rápido de construir;
- explicação por exemplos semelhantes é útil;
- latência não é extremamente restrita.

## 14.3 Quando reconsiderar

- milhões de amostras com busca exata;
- centenas/milhares de atributos sem estrutura;
- inferência de baixíssima latência;
- memória limitada;
- necessidade de extrapolação;
- distância sem interpretação clara;
- forte drift de dados;
- exigência de um modelo global compacto.

---

# Parte XV — Pegadinhas presentes nas questões

1. **KNN não é K-Means.** KNN é supervisionado; K-Means é agrupamento.
2. **KNN é preditivo.** Ele prediz classes ou valores.
3. **KNN serve para classificação e regressão.**
4. **KNN suporta multiclasse.**
5. **Não paramétrico não significa sem hiperparâmetros.**
6. **`fit` não é literalmente “não fazer nada”.** Pode validar, armazenar e indexar.
7. **K é o número de vizinhos, não o número de grupos.**
8. **K pequeno aumenta variância e risco de overfitting.**
9. **K grande aumenta viés e risco de underfitting.**
10. **Com K=N e voto uniforme, vence a classe global majoritária.**
11. **`weights='distance'` não escolhe K automaticamente.**
12. **K ímpar não elimina todos os empates.**
13. **`KNeighborsClassifier` não possui parâmetro `priority`.**
14. **Empate não cria uma classe `tie`.**
15. **Escala é geralmente importante, mas não universalmente obrigatória.**
16. **Escalonar não aumenta necessariamente a velocidade de cálculo.** Corrige a geometria.
17. **`StandardScaler` produz média 0 e variância 1**, salvo colunas constantes e detalhes numéricos.
18. **`MinMaxScaler` não binariza dados na presença de outliers.**
19. **Scaler deve ser ajustado somente no treino.**
20. **Outlier não deve ser excluído automaticamente.** Primeiro se investiga.
21. **Peso por distância não resolve todos os outliers.** Um ruído próximo pode ganhar muito peso.
22. **Logaritmo nos atributos não resolve desbalanceamento de classes.**
23. **Hamming, Mahalanobis, Manhattan e Euclidiana podem ser usadas**, desde que adequadas à representação e ao backend.
24. **Minkowski com p=1 é Manhattan; p=2 é Euclidiana.**
25. **Manhattan não é automaticamente melhor em alta dimensão.**
26. **Mahalanobis exige covariância estável/invertível ou regularizada.**
27. **Custo brute force é O(ND) por consulta**, não apenas “linear” sem contexto.
28. **KDTree não é sempre mais rápida que brute force.**
29. **KDTree piora em alta dimensão.**
30. **`leaf_size` menor não é sempre melhor.**
31. **BallTree não foi feita para corrigir KDTree em baixa dimensão.**
32. **RadiusNeighbors usa raio, não K fixo.**
33. **Probabilidade local do KNN não é garantia de calibração.**
34. **Erro de treino mínimo não garante melhor generalização.**

---

# Parte XVI — Respostas-modelo para falar com segurança

## “Como o KNN funciona?”

> O KNN mede a distância entre uma nova observação e os exemplos do treino, seleciona os K mais próximos e agrega suas respostas. Na classificação usa votação; na regressão, média. Seu sucesso depende de a distância representar similaridade relevante para o alvo.

## “KNN é supervisionado e preditivo?”

> Sim. Ele usa exemplos com alvo conhecido e prediz respostas para novos registros. É também baseado em instâncias, lazy e não paramétrico.

## “KNN treina?”

> Ele não aprende coeficientes, mas o `fit` valida e armazena os dados e pode construir um índice de busca. O ajuste é relativamente barato e o trabalho pesado tende a ocorrer na predição.

## “O que acontece quando K aumenta?”

> A fronteira fica mais suave: a variância diminui e o viés aumenta. Isso reduz sensibilidade a ruído, mas K excessivo causa underfitting e favorece a classe majoritária. Eu escolheria K por validação cruzada.

## “Por que KNN precisa de escala?”

> Porque sua decisão depende de distâncias. Um atributo com magnitude maior pode dominar a medida sem ser mais importante. Ajustaria o scaler apenas no treino, dentro de um pipeline, e validaria scaler, métrica e K em conjunto.

## “Qual métrica usar?”

> Depende do tipo de dado e do significado de proximidade. Euclidiana e Manhattan são comuns em dados numéricos; Hamming pode servir para representações binárias/categóricas; Mahalanobis incorpora correlações, mas exige covariância estável. A escolha deve ser validada.

## “Como lidar com outliers?”

> Primeiro investigo se são erro, evento raro legítimo ou mudança de regime. Depois comparo estratégias como correção, remoção justificada, RobustScaler, outra métrica, K maior ou pesos. Peso por distância ajuda contra pontos distantes, mas pode amplificar ruído muito próximo.

## “Como lidar com desbalanceamento?”

> KNN pode favorecer a maioria porque ela aparece mais nas vizinhanças. Eu avaliaria reamostragem dentro do treino, pesos por distância e classe, escolha de K e ajuste de limiar, usando recall, F1, PR-AUC ou balanced accuracy em vez de apenas acurácia.

## “Qual é o custo computacional?”

> Em brute force, uma consulta custa aproximadamente O(ND), pois compara o ponto com N amostras em D dimensões. Para Q consultas, O(QND), além da seleção dos vizinhos. KDTree ou BallTree podem acelerar em baixa ou moderada dimensão, mas perdem eficiência com a maldição da dimensionalidade.

## “KDTree ou BallTree?”

> KDTree particiona o espaço por eixos e costuma ser eficiente em baixa dimensão. BallTree usa regiões definidas por centro e raio e suporta outras métricas e geometrias. A melhor opção depende de N, D, métrica e hardware; em alta dimensão, brute force pode vencer ambas.

## “Quais são os parâmetros e hiperparâmetros?”

> O estado ajustado inclui dados, alvos e possivelmente um índice. Os principais hiperparâmetros são K, pesos, métrica, p, algoritmo de busca e leaf_size. KNN ser não paramétrico não significa não possuir hiperparâmetros.

---

# Parte XVII — Roteiro de treinamento

## Sessão 1 — Classificação e natureza do KNN

- diferenciar classificação, regressão e clustering;
- explicar supervisionado, preditivo, lazy, instância e não paramétrico;
- simular uma predição manual.

Critério de domínio: explicar o algoritmo do dado bruto até a classe final.

## Sessão 2 — K, votação e viés–variância

- variar K e desenhar fronteiras mentais;
- comparar `uniform` e `distance`;
- resolver empates;
- explicar K=1 e K=N.

Critério de domínio: prever efeitos em treino, validação, viés e variância.

## Sessão 3 — Distâncias e escala

- calcular Euclidiana, Manhattan e Minkowski;
- interpretar Hamming, Chebyshev e Mahalanobis;
- comparar scalers;
- montar pipeline sem vazamento.

Critério de domínio: escolher e justificar métrica e scaler para três conjuntos diferentes.

## Sessão 4 — Robustez

- analisar outliers;
- diagnosticar desbalanceamento;
- escolher métricas de avaliação;
- discutir dimensionalidade e atributos irrelevantes.

Critério de domínio: propor mitigação sem apagar dados ou usar acurácia automaticamente.

## Sessão 5 — Custo e estruturas de busca

- derivar O(ND) por consulta;
- comparar brute, KDTree e BallTree;
- explicar `leaf_size`;
- relacionar alta dimensão à perda de eficiência.

Critério de domínio: escolher backend considerando N, D, métrica, memória e latência.

---

# Checklist final de domínio

Você domina o conteúdo quando consegue, sem consultar material:

- definir classificação e diferenciar regressão e clustering;
- explicar por que KNN é supervisionado, preditivo, lazy, baseado em instâncias e não paramétrico;
- descrever `fit` e `predict` sem afirmar que treino é inexistente;
- simular votação uniforme e ponderada;
- explicar K=1, K grande e K=N;
- relacionar K a viés, variância, overfitting e underfitting;
- calcular Euclidiana, Manhattan e Minkowski;
- interpretar Chebyshev, Hamming e Mahalanobis;
- justificar escala e escolher um scaler;
- evitar vazamento com pipeline;
- diagnosticar outliers antes de removê-los;
- tratar desbalanceamento com métricas e amostragem adequadas;
- derivar custo O(ND) por consulta;
- comparar brute force, KDTree, BallTree e busca aproximada;
- explicar maldição da dimensionalidade;
- diferenciar KNeighborsClassifier de RadiusNeighborsClassifier;
- separar estado aprendido de hiperparâmetros;
- reconhecer e corrigir todas as pegadinhas listadas.