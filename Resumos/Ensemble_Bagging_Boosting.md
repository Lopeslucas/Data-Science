# Guia de domínio — Ensemble, Bagging, Random Forest e Boosting

Material consolidado a partir das perguntas da lista sobre **Ensemble Learning, viés–variância, Bagging, Random Forest, Boosting e AdaBoost**.

O objetivo não é apenas memorizar alternativas. Uma resposta segura deve percorrer:

> **definição → mecanismo → efeito sobre viés/variância → custo → limitação → forma de validação**

---

# Parte I — Fundamentos de Ensemble Learning

## 1.1 O que é um ensemble?

Um **ensemble** combina as previsões de vários modelos-base para produzir uma previsão final. A ideia é que modelos com erros parcialmente diferentes podem, quando combinados corretamente, gerar um sistema mais preciso e robusto que cada modelo isolado.

Um ensemble não garante melhoria automaticamente. O ganho depende de três fatores:

- os modelos-base terem alguma capacidade preditiva;
- existir diversidade entre seus erros;
- a regra de combinação ser adequada.

Se todos os modelos cometem os mesmos erros, combiná-los acrescenta custo sem reduzir muito o erro.

## 1.2 Natureza e tarefas

Na forma estudada nesta lista, ensembles são:

- **supervisionados**, pois aprendem com exemplos rotulados;
- **preditivos**, pois estimam alvos para novos registros;
- aplicáveis à **classificação** e à **regressão**.

Não se deve definir ensemble como uma técnica de agrupamento. Existem ensembles em outros contextos, inclusive clustering, mas Bagging, Random Forest e AdaBoost são normalmente apresentados como métodos supervisionados.

## 1.3 Homogêneo e heterogêneo

- **Homogêneo:** repete o mesmo tipo de modelo-base, como várias árvores em uma Random Forest.
- **Heterogêneo:** combina algoritmos diferentes, como árvore, regressão logística e KNN em voting ou stacking.

Um ensemble pode variar modelos, dados, atributos, hiperparâmetros, sementes aleatórias ou várias dessas dimensões.

## 1.4 Principais famílias

- **Voting/Averaging:** combina diretamente votos, probabilidades ou valores previstos.
- **Bagging:** treina modelos independentes em amostras diferentes e agrega suas respostas.
- **Random Forest:** Bagging de árvores com aleatoriedade adicional nos atributos candidatos a cada divisão.
- **Boosting:** treina modelos sequencialmente, fazendo cada etapa concentrar-se nas dificuldades das anteriores.
- **Stacking:** usa as previsões dos modelos-base como entradas de um meta-modelo.

Bagging, Boosting e Stacking são técnicas de ensemble. Leave-One-Out e Leave-P-Out são validações; one-vs-rest e one-vs-one são estratégias de decomposição multiclasse, não famílias de ensemble por si mesmas.

## 1.5 Como as previsões são combinadas

### Classificação

- **hard voting:** classe mais votada, isto é, a moda;
- **soft voting:** média, eventualmente ponderada, das probabilidades por classe;
- **voto ponderado:** modelos com maior peso influenciam mais a decisão.

### Regressão

- média simples;
- média ponderada;
- em aplicações específicas, mediana ou outra agregação robusta.

Soft voting exige probabilidades comparáveis. Se modelos produzem probabilidades mal calibradas, a média pode parecer confiante sem ser confiável.

---

# Parte II — Viés, variância, overfitting e underfitting

## 2.1 Viés

Viés é o erro sistemático causado por hipóteses simplificadoras do método. Um modelo de **alto viés** tende a não capturar adequadamente a estrutura do problema.

Não é correto definir viés simplesmente como “erro no treino”. Erro de treino alto pode ser um indício de alto viés, mas os conceitos não são equivalentes.

## 2.2 Variância

Variância é a sensibilidade do modelo a mudanças na amostra de treinamento. Um modelo de **alta variância** muda muito quando treinado em conjuntos ligeiramente diferentes.

A diferença entre erro de treino e validação/teste é um indício prático de overfitting, mas não é a definição matemática de variância.

## 2.3 Overfitting e underfitting

### Overfitting

O modelo ajusta detalhes e ruído do treino, alcança erro de treino muito baixo e generaliza mal. Está associado, em geral, a complexidade excessiva e alta variância.

### Underfitting

O modelo é simples ou restrito demais para capturar o padrão. Apresenta desempenho fraco inclusive no treino e costuma estar associado a alto viés.

## 2.4 Interpretação do alvo de viés e variância

- tiros concentrados no centro: baixo viés e baixa variância;
- tiros dispersos ao redor do centro: baixo viés e alta variância;
- tiros concentrados longe do centro: alto viés e baixa variância;
- tiros dispersos longe do centro: alto viés e alta variância.

## 2.5 Complexidade do modelo

À medida que a complexidade aumenta:

- o erro de treino tende a cair;
- o erro de validação/teste tende a cair inicialmente e depois pode subir;
- o melhor ponto é escolhido pelo menor erro de validação esperado, não pela igualdade entre erros de treino e teste.

O conjunto de teste não deve orientar a escolha da complexidade. Ele deve ser preservado para a avaliação final.

## 2.6 Onde entram os ensembles?

- **Bagging** atua principalmente na redução de variância.
- **Boosting** historicamente é descrito como redutor de viés, embora também possa afetar a variância e sofrer overfitting.
- **Random Forest** reduz variância ao agregar árvores e diminuir sua correlação.

Essas são tendências, não garantias universais.

---

# Parte III — Bagging

## 3.1 Significado e mecanismo

**Bagging** significa **Bootstrap Aggregating**.

Procedimento típico:

1. parta de um treino com `N` observações;
2. gere `B` amostras bootstrap;
3. treine um modelo-base em cada amostra;
4. produza previsões independentes;
5. agregue por voto/moda na classificação ou média na regressão.

Os modelos podem ser treinados em paralelo porque um não depende do resultado do outro.

## 3.2 Bootstrap

Uma amostra bootstrap é obtida por amostragem **com reposição**. Em sua forma clássica, sorteiam-se `N` elementos de um conjunto com `N` registros.

Consequências:

- alguns registros aparecem mais de uma vez;
- outros não aparecem;
- a amostra tem `N` sorteios, mas menos de `N` observações únicas.

Para `N` grande, a fração esperada de observações únicas é:

\[
1-\left(1-\frac{1}{N}\right)^N \approx 1-e^{-1}\approx 63{,}2\%
\]

Logo, aproximadamente `36,8%` das observações ficam fora de uma amostra bootstrap. É errado dizer que `36,8%` são necessariamente “duplicadas”; esse percentual descreve melhor a fração **não selecionada**.

## 3.3 Out-of-bag (OOB)

As observações não incluídas na amostra de uma unidade do ensemble são **out-of-bag** para aquela unidade.

Para estimar a previsão OOB de um registro:

1. selecione apenas os modelos para os quais esse registro ficou fora do bootstrap;
2. agregue suas previsões;
3. compare com o alvo verdadeiro.

O resultado fornece uma estimativa interna de generalização e pode reduzir a necessidade de um conjunto de validação separado. Contudo:

- OOB não substitui o teste final;
- não deve haver pré-processamento com vazamento antes do ensemble;
- cada registro é avaliado por apenas uma parte dos modelos;
- poucos estimadores tornam a estimativa OOB mais instável.

No Bagging padrão, OOB não significa excluir automaticamente os estimadores com baixo desempenho. Essa seria uma modificação do algoritmo e exigiria validação independente para não introduzir viés de seleção.

## 3.4 Por que reduz variância?

Ao fazer a média de `B` estimadores com variância aproximada `σ²` e correlação média `ρ`, a variância da média pode ser interpretada aproximadamente como:

\[
Var(\bar f) \approx \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
\]

Ao aumentar `B`, a segunda parcela diminui, mas a parcela causada pela correlação permanece. Portanto:

- mais estimadores estabilizam a previsão;
- modelos muito correlacionados limitam o ganho;
- diversidade é tão importante quanto quantidade.

## 3.5 Quando funciona melhor

Bagging costuma ser especialmente útil com modelos **instáveis e de alta variância**, como árvores profundas. Pequenas mudanças nos dados produzem árvores diferentes, e a agregação estabiliza essas mudanças.

Bagging pode ser aplicado a outros estimadores, mas traz menos benefício quando o modelo-base já é muito estável ou quando todos os modelos aprendem praticamente a mesma função.

## 3.6 Escala, outliers e desbalanceamento

### Escala

Bagging não define sozinho se é preciso escalonar. Isso depende do modelo-base:

- árvore: geralmente não precisa;
- KNN, SVM ou modelo sensível a distâncias: pode precisar;
- transformações devem ser ajustadas sem vazamento, idealmente em `Pipeline`.

### Outliers

Outliers podem ser repetidos em algumas amostras bootstrap e influenciar fortemente certos modelos. O impacto depende do estimador-base e da função de perda. Bagging pode diluir influências localizadas, mas não torna o sistema imune.

### Desbalanceamento

O bootstrap comum preserva o desbalanceamento em expectativa e pode gerar amostras com pouquíssimos representantes da classe rara. Possíveis respostas:

- bootstrap estratificado ou balanceado;
- `class_weight` ou pesos de amostra;
- Balanced Random Forest;
- reamostragem somente dentro do treino;
- métricas como recall, F1, PR-AUC e balanced accuracy.

## 3.7 Pontos positivos

- reduz variância e instabilidade;
- melhora a robustez de estimadores instáveis;
- permite paralelização do treino e da predição;
- funciona em classificação e regressão;
- possibilita avaliação OOB quando há bootstrap;
- costuma exigir menos ajuste fino que métodos de boosting.

## 3.8 Pontos negativos e como lidar

- **Maior custo e memória:** limitar `n_estimators`, profundidade, amostra e atributos; paralelizar.
- **Menor interpretabilidade:** usar importância por permutação, explicações locais e modelos substitutos com cautela.
- **Modelos correlacionados:** aleatorizar atributos, amostras ou hiperparâmetros; Random Forest faz isso explicitamente.
- **Desbalanceamento:** pesos, amostragem balanceada e métricas adequadas.
- **Outliers:** investigar dados, usar estimadores/perdas robustas e validar o impacto.
- **Retorno decrescente de mais modelos:** acompanhar curva OOB/validação versus `n_estimators`.

## 3.9 Custo computacional

Se treinar um modelo-base custa `T_train(N,D)` e predizer custa `T_pred(D)`, para `B` modelos:

\[
T_{treino}\approx B\cdot T_{train}(M,D)
\]

\[
T_{predição}\approx B\cdot T_{pred}(D)
\]

em que `M` é o tamanho de cada amostra. A agregação adiciona aproximadamente `O(B)` por exemplo.

O tempo de parede pode cair com paralelização, mas o trabalho computacional total e a memória ainda crescem aproximadamente com `B`. Por isso, não existe uma única resposta “linear”, “quadrática” ou “logarítmica” sem especificar o estimador-base.

---

# Parte IV — Random Forest

## 4.1 Como funciona

Random Forest é um ensemble homogêneo de árvores que combina:

1. amostras bootstrap dos registros, normalmente;
2. treinamento de uma árvore por amostra;
3. seleção aleatória de um subconjunto de atributos **em cada nó/corte**;
4. agregação das árvores por voto ou média.

A aleatoriedade nos atributos reduz a correlação entre árvores. Isso distingue Random Forest de um Bagging simples de árvores.

Não é preciso que cada árvore seja de um “tipo diferente”. Em geral, todas usam a mesma família e configuração, variando dados, atributos candidatos e semente.

## 4.2 `max_features`

`max_features` determina quantos atributos são candidatos em cada divisão.

- valor menor: mais diversidade e menor correlação, mas splits individuais potencialmente mais fracos;
- valor maior: árvores individualmente mais fortes, porém mais correlacionadas.

Regras como `sqrt(D)` são defaults/heurísticas comuns em classificação, não leis universais. O valor depende da biblioteca, versão e tarefa e deve ser validado.

## 4.3 Classificação e regressão

- **Classificação:** voto das classes ou agregação de probabilidades, conforme a implementação.
- **Regressão:** média das previsões das árvores.

É um método supervisionado, preditivo, não paramétrico e eager.

## 4.4 Dados numéricos e categóricos

Conceitualmente, árvores podem tratar variáveis numéricas e categóricas. A implementação concreta pode exigir codificação.

No `scikit-learn`, a entrada das Random Forests tradicionais é numérica; categorias precisam ser representadas de forma compatível. Codificação ordinal arbitrária pode criar ordens artificiais, portanto a estratégia deve ser escolhida com cuidado.

## 4.5 Escala, outliers e desbalanceamento

### Escala

Random Forest não depende de distância e geralmente não exige normalização ou padronização. Transformações monotônicas preservam a ordem usada nos cortes.

### Outliers

Não é imune. Outliers podem alterar limiares, formar folhas pequenas e, em regressão com média/erro quadrático, exercer influência relevante. Limitar profundidade, aumentar `min_samples_leaf`, usar critérios robustos quando disponíveis e revisar dados pode ajudar.

### Desbalanceamento

Pode favorecer a classe majoritária. Use:

- `class_weight='balanced'` ou `'balanced_subsample'` quando apropriado;
- amostragem balanceada;
- ajuste de limiar;
- métricas alinhadas ao custo do erro;
- validação cruzada estratificada.

## 4.6 Critérios de divisão

Em classificação, critérios comuns incluem:

- índice de Gini;
- entropia;
- log loss, conforme a implementação.

Gini e entropia medem impureza. Dunn e Davies–Bouldin são métricas de clustering. “CART” é uma família de algoritmo, não um índice de impureza; Breiman é um pesquisador associado ao CART e Random Forest, não um critério chamado “índice de Breiman”.

## 4.7 Parâmetros versus hiperparâmetros

### Parâmetros aprendidos

São determinados durante o ajuste:

- atributos e limiares dos cortes;
- estrutura de cada árvore;
- classes/valores e distribuições nas folhas.

### Hiperparâmetros definidos antes do ajuste

- `n_estimators`;
- `max_features`;
- `max_depth`;
- `min_samples_split`;
- `min_samples_leaf`;
- `max_leaf_nodes`;
- `criterion`;
- `bootstrap` e `max_samples`;
- `class_weight`;
- `ccp_alpha`;
- `random_state`;
- `n_jobs` — operacional, não controla a função aprendida diretamente.

“Não paramétrico” não significa “sem parâmetros aprendidos” nem “sem hiperparâmetros”. Significa que a complexidade efetiva não é fixada por um vetor de dimensão constante independente dos dados.

## 4.8 Efeito de `n_estimators`

Aumentar o número de árvores normalmente:

- reduz a variabilidade Monte Carlo do ensemble;
- estabiliza desempenho e importância;
- aumenta tempo, memória e latência;
- apresenta retorno decrescente.

Mais árvores, isoladamente, geralmente não fazem uma Random Forest sofrer overfitting da mesma forma que aumentar a profundidade de uma árvore. O erro costuma estabilizar. Porém, vazamento, ajuste excessivo de hiperparâmetros ou árvores muito correlacionadas continuam sendo problemas.

## 4.9 Pontos positivos

- bom desempenho tabular como baseline;
- reduz a variância de árvores isoladas;
- captura não linearidades e interações;
- não exige escala na maioria dos casos;
- tolera muitos atributos e oferece importância de variáveis;
- classificação e regressão;
- treino paralelizável;
- estimativa OOB disponível quando configurada.

## 4.10 Pontos negativos e como lidar

- **Menos interpretável que uma árvore:** usar importância por permutação, PDP/ICE ou explicações locais; não tratar importância como causalidade.
- **Grande em memória e latência:** reduzir árvores/profundidade, usar `max_samples`, compressão ou destilação.
- **Importância por impureza enviesada:** preferir importância por permutação em dados de validação e considerar correlação entre atributos.
- **Má extrapolação em regressão:** árvores predizem combinações de valores vistos nas folhas; considerar modelos com estrutura extrapolativa.
- **Desbalanceamento:** pesos, reamostragem e limiar.
- **Correlação entre árvores:** ajustar `max_features`, amostra e diversidade.
- **Probabilidades pouco calibradas:** avaliar Brier/log loss e aplicar calibração fora do treino quando necessário.

## 4.11 Custo computacional

Para `B` árvores, `N` registros, `D` atributos e `m` atributos candidatos por nó, uma aproximação comum para treino balanceado é:

\[
O(B\,m\,N\log N)
\]

Essa expressão varia conforme ordenação, profundidade, amostragem e implementação. Com `M=max_samples`, substitui-se `N` por `M` por árvore.

Predição para um exemplo:

\[
O(Bh)
\]

onde `h` é a profundidade média. A memória é aproximadamente proporcional ao total de nós de todas as árvores.

Random Forest geralmente custa mais que uma única árvore porque constrói várias. Comparações absolutas com KNN e Naive Bayes dependem de se a pergunta trata treino, predição, `N`, `D` e implementação.

---

# Parte V — Boosting

## 5.1 Ideia central

Boosting constrói um modelo forte por adição sequencial de modelos-base. Cada novo estimador é treinado para melhorar o conjunto já existente.

Diferentes famílias fazem isso de maneiras diferentes:

- **AdaBoost:** aumenta a influência das observações mal classificadas e pondera os estimadores;
- **Gradient Boosting:** ajusta novos modelos aos gradientes negativos/resíduos da função de perda;
- **XGBoost, LightGBM e CatBoost:** implementações/famílias otimizadas com regularização e estratégias próprias.

Portanto, “Boosting sempre reamostra erros” é uma simplificação aplicável a certas interpretações do AdaBoost, não a toda a família.

## 5.2 Sequencialidade

Ao contrário do Bagging, os estágios dependem dos anteriores. Isso reduz a paralelização direta entre estimadores, embora implementações possam paralelizar cálculos internos.

A predição final usa o conjunto de estágios. Não se usa apenas o último estimador.

## 5.3 Modelos-base

Árvores rasas são muito comuns, mas Boosting não é restrito a árvores nem exige sempre `max_depth=1`.

- um stump é uma árvore com uma divisão, frequentemente usada no AdaBoost clássico;
- profundidades maiores podem ser adequadas;
- o estimador deve aceitar pesos de amostra quando o algoritmo depende deles.

## 5.4 Pontos positivos

- pode reduzir fortemente viés e erro preditivo;
- combina modelos simples em uma fronteira complexa;
- captura não linearidades e interações com árvores;
- oferece funções de perda e regularização flexíveis;
- costuma alcançar excelente desempenho em dados tabulares.

## 5.5 Pontos negativos e como lidar

- **Treino sequencial e mais lento:** early stopping, menos estágios, histogramas/amostragem e implementação otimizada.
- **Sensibilidade a ruído/outliers:** perdas robustas, limpeza criteriosa, regularização, menor profundidade e taxa de aprendizado.
- **Overfitting possível:** validação, early stopping, `learning_rate`, profundidade, subsampling e regularização.
- **Muitos hiperparâmetros:** busca orientada por validação e limites de orçamento.
- **Menor interpretabilidade:** explicações globais/locais com cautela e comparação com baseline simples.
- **Desbalanceamento:** pesos de classe/amostra, métricas adequadas e ajuste de limiar.

## 5.6 Escala

Escala depende do estimador-base. Boosting de árvores normalmente não exige padronização. Um boosting baseado em estimadores sensíveis à escala herda essa sensibilidade.

Em outras palavras, as limitações do ensemble incluem limitações dos modelos-base, mas a combinação também cria propriedades novas.

## 5.7 Custo computacional

Para `B` estágios:

\[
T_{treino}\approx \sum_{b=1}^{B}T_{base,b}+T_{atualização,b}
\]

Se os estágios têm custo semelhante, o trabalho cresce aproximadamente de forma linear em `B`, mas é principalmente sequencial.

Predição também percorre os `B` estimadores. Em boosting de árvores:

\[
T_{predição}\approx O(Bh)
\]

com `h` representando a profundidade média. A complexidade exata depende da variante e do estimador-base.

---

# Parte VI — AdaBoost

## 6.1 Como funciona

No AdaBoost binário clássico:

1. inicialize todas as observações com o mesmo peso;
2. treine um estimador fraco usando esses pesos;
3. calcule seu erro ponderado;
4. atribua maior peso de voto a estimadores melhores;
5. aumente a influência relativa dos exemplos errados;
6. normalize os pesos;
7. repita;
8. combine os estimadores por voto ponderado.

## 6.2 Pesos iniciais

Para `N` observações:

\[
w_i^{(1)}=\frac{1}{N}
\]

Assim, os pesos somam 1.

## 6.3 Erro ponderado

Para o estimador `t`:

\[
\epsilon_t=\sum_{i=1}^{N}w_i^{(t)}\,\mathbb{1}[y_i\neq h_t(x_i)]
\]

## 6.4 Peso do estimador

Uma convenção comum para rótulos `{-1,+1}` é:

\[
\alpha_t=\frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)
\]

Alguns materiais usam:

\[
\alpha_t=\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)
\]

e compensam o fator na atualização dos pesos. As duas formas podem representar convenções equivalentes quando todo o algoritmo é formulado de modo consistente.

Interpretação:

- erro baixo → `α` grande e positivo;
- erro próximo de `0,5` → pouca ou nenhuma contribuição no caso binário;
- erro maior que `0,5` → `α` negativo na fórmula clássica, indicando um estimador pior que o acaso; implementações podem rejeitar, inverter ou interromper.

## 6.5 Atualização dos pesos das observações

Forma compacta, com `y_i,h_t(x_i)∈{-1,+1}`:

\[
w_i^{(t+1)}=\frac{w_i^{(t)}\exp[-\alpha_t y_i h_t(x_i)]}{Z_t}
\]

em que `Z_t` normaliza a soma para 1.

Consequência:

- classificação correta: multiplica por `e^{-α_t}`;
- classificação incorreta: multiplica por `e^{+α_t}`.

Algumas derivações mantêm os errados e reduzem apenas os corretos antes da normalização. O efeito relativo é equivalente dentro da convenção adotada. Não se deve misturar fórmulas de convenções diferentes.

## 6.6 Predição final

No caso binário clássico:

\[
H(x)=sign\left(\sum_{t=1}^{B}\alpha_t h_t(x)\right)
\]

Ou seja, soma-se o peso dos estimadores que votam em cada classe e escolhe-se a classe com maior soma. Não é voto simples e não é apenas a resposta do último estimador.

## 6.7 Classificação multiclasse e regressão

AdaBoost não se limita à classificação binária. Existem extensões como SAMME/SAMME.R para multiclasse e AdaBoost.R2 para regressão. As fórmulas diferem do caso binário original.

## 6.8 Parâmetros versus hiperparâmetros

### Parâmetros aprendidos

- parâmetros internos de cada estimador-base;
- erros ponderados `ε_t`;
- pesos dos estimadores `α_t`;
- evolução dos pesos das observações durante o ajuste.

### Hiperparâmetros

- `estimator` e seus hiperparâmetros;
- `n_estimators`;
- `learning_rate`;
- algoritmo/variante, quando exposto;
- `random_state`;
- parâmetros de profundidade e regularização do modelo-base.

AdaBoost com árvores continua sendo normalmente descrito como não paramétrico. Ter `α`, pesos e hiperparâmetros não o torna paramétrico no sentido estatístico clássico.

## 6.9 Outliers e ruído de rótulo

AdaBoost dá atenção crescente aos erros. Se um ponto é um outlier ou possui rótulo incorreto, ele pode receber peso crescente e desviar os estágios seguintes.

Mitigações:

- revisar rótulos e qualidade dos dados;
- limitar `n_estimators`;
- reduzir `learning_rate`;
- regularizar o modelo-base;
- avaliar variantes/perdas mais robustas;
- usar validação e acompanhar as curvas de treino e validação.

## 6.10 Desbalanceamento

AdaBoost não resolve automaticamente o desbalanceamento. Erros da classe minoritária podem ganhar peso, mas isso não garante boa cobertura nem alinhamento com o custo do problema.

Use pesos iniciais/custos por classe quando suportados, reamostragem dentro do treino, métricas adequadas e ajuste do limiar.

---

# Parte VII — Bagging versus Boosting

| Aspecto | Bagging | Boosting |
|---|---|---|
| Organização | Modelos independentes | Modelos dependentes e sequenciais |
| Dados | Bootstrap/amostras aleatórias | Pesos, resíduos ou gradientes adaptativos |
| Agregação | Voto ou média, normalmente uniforme | Soma/voto ponderado ou soma aditiva |
| Objetivo típico | Reduzir variância | Reduzir viés/erro aditivo |
| Modelo-base comum | Forte e instável, como árvore profunda | Fraco/regularizado, como árvore rasa |
| Paralelização | Alta entre modelos | Limitada entre estágios |
| Ruído/outliers | Pode diluir, mas não é imune | Pode enfatizar casos difíceis/ruidosos |
| Mais estimadores | Estabiliza e tende a saturar | Pode continuar ajustando e overfit é possível |
| Exemplos | BaggingClassifier, Random Forest | AdaBoost, Gradient Boosting |

“Bagging só reduz variância” e “Boosting só reduz viés” são atalhos pedagógicos. O efeito real depende dos dados, modelos-base, perda e regularização.

---

# Parte VIII — Premissas e critérios para usar

## 8.1 Premissas gerais de ensembles

Não há uma hipótese distribucional única, como normalidade obrigatória dos atributos. As premissas práticas são:

- os dados futuros devem ser suficientemente semelhantes ao regime de treino;
- o alvo e a métrica devem refletir o objetivo do negócio;
- a validação deve evitar vazamento;
- os modelos-base precisam ser melhores que uma referência ingênua ou contribuir com diversidade útil;
- o ganho preditivo deve justificar custo, memória, latência e perda de interpretabilidade.

## 8.2 Quando preferir Bagging/Random Forest

- uma árvore isolada apresenta alta variância;
- há dados tabulares com relações não lineares;
- deseja-se um baseline forte com pouco pré-processamento;
- treino paralelo é possível;
- robustez é mais importante que interpretação de uma única regra;
- OOB é útil como diagnóstico interno.

## 8.3 Quando preferir Boosting

- busca-se alto desempenho preditivo em dados tabulares;
- modelos simples sofrem underfitting;
- há orçamento para ajuste de hiperparâmetros;
- latência e treinamento sequencial são aceitáveis;
- ruído/outliers estão razoavelmente controlados;
- há validação confiável e, idealmente, early stopping.

## 8.4 Quando evitar ou reconsiderar

- explicabilidade regulatória exige regras simples e diretas;
- latência/memória são extremamente restritas;
- conjunto muito pequeno não sustenta validação confiável;
- mudança de distribuição domina o problema;
- os dados têm vazamento ou rótulos ruins;
- um modelo simples já satisfaz a métrica e o custo.

---

# Parte IX — Como configurar e validar

## 9.1 Fluxo recomendado

1. defina alvo, métrica e custo dos erros;
2. separe teste final antes de explorar;
3. construa um baseline simples;
4. coloque pré-processamento dentro de pipeline/folds;
5. compare Bagging, Random Forest e Boosting com validação apropriada;
6. ajuste poucos hiperparâmetros de maior impacto;
7. analise treino versus validação e curvas de aprendizado;
8. avalie calibração, subgrupos, classe rara, latência e memória;
9. realize uma única avaliação final no teste;
10. monitore drift e desempenho em produção.

## 9.2 Hiperparâmetros prioritários

### Bagging

- `estimator`;
- `n_estimators`;
- `max_samples`;
- `max_features`;
- `bootstrap`/`bootstrap_features`;
- complexidade do estimador-base.

### Random Forest

- `n_estimators`;
- `max_features`;
- `max_depth`;
- `min_samples_leaf`;
- `max_samples`;
- `class_weight`;
- `criterion`.

### AdaBoost

- `estimator` e sua complexidade;
- `n_estimators`;
- `learning_rate`.

## 9.3 Relações importantes

- mais `n_estimators` aumenta custo em todos os métodos;
- em Random Forest, mais árvores reduz instabilidade até saturar;
- em Boosting, `learning_rate` menor costuma exigir mais estimadores;
- árvores mais profundas reduzem viés, mas elevam complexidade e risco de overfitting;
- `max_features` menor aumenta diversidade na Random Forest, mas pode enfraquecer cada árvore;
- `min_samples_leaf` maior suaviza árvores e pode aumentar robustez.

---

# Parte X — Pegadinhas presentes nas questões

1. **Ensemble não é simplesmente “usar muitos modelos”.** É preciso combinar suas previsões.
2. **Mais modelos não garantem melhor resultado.** Erros altamente correlacionados limitam o ganho.
3. **Bagging significa Bootstrap Aggregating.**
4. **Bootstrap é com reposição.** Sem reposição é outra forma de subamostragem.
5. **A amostra bootstrap clássica tem `N` sorteios, não `N` exemplos únicos.**
6. **Os `36,8%` referem-se aproximadamente aos exemplos fora da amostra**, não a uma regra de “36,8% duplicados”.
7. **OOB é relativo a cada modelo.** Um exemplo pode estar dentro do bootstrap de uma árvore e fora do de outra.
8. **OOB estima o ensemble por votos das árvores que não viram cada registro**, não apenas o desempenho isolado de uma árvore.
9. **Bagging não exige escala por si mesmo.** A necessidade vem do estimador-base.
10. **Bagging não resolve desbalanceamento automaticamente.**
11. **Random Forest não é apenas muitas árvores.** Também sorteia atributos candidatos em cada corte.
12. **`sqrt(D)` para `max_features` é heurística/default possível, não premissa universal.**
13. **Random Forest normalmente não precisa de scaler.**
14. **Random Forest não é imune a outliers ou desbalanceamento.**
15. **Gini e entropia são critérios de impureza; Dunn e Davies–Bouldin são de clustering.**
16. **CART não é um índice.** É uma família de árvores.
17. **Aumentar apenas `n_estimators` da Random Forest tende a estabilizar, não a causar overfitting inevitável.**
18. **Boosting é sequencial; Bagging é independente/paralelizável.**
19. **Boosting não é restrito a árvores nem a stumps.**
20. **AdaBoost não usa somente o último modelo.** Usa voto/soma ponderada de todos.
21. **AdaBoost não usa voto simples de maioria.** Cada estimador recebe peso `α`.
22. **A fórmula de `α` pode ter ou não o fator `1/2`**, dependendo da convenção completa.
23. **AdaBoost enfatiza erros, inclusive ruído e rótulos incorretos.**
24. **Não paramétrico não significa sem parâmetros ou hiperparâmetros.**
25. **Complexidade computacional depende do modelo-base, número de estimadores e fase analisada.**
26. **Erro de teste não deve escolher hiperparâmetros.** Use validação; teste é avaliação final.
27. **Viés não é simplesmente erro de treino e variância não é simplesmente a diferença treino–teste.**
28. **O melhor modelo não é necessariamente onde treino e teste têm erros iguais.** É o que minimiza erro de generalização sob a métrica relevante.

---

# Parte XI — Respostas-modelo para falar com segurança

## “O que é Ensemble Learning?”

> É a estratégia de combinar previsões de vários modelos-base para obter uma decisão final potencialmente mais precisa e robusta. O ganho ocorre quando os modelos têm capacidade preditiva e cometem erros suficientemente diversos; combinar modelos iguais nos mesmos erros apenas aumenta custo.

## “Como funciona o Bagging?”

> Bagging cria várias amostras bootstrap com reposição, treina modelos independentes e agrega suas previsões por voto na classificação ou média na regressão. Seu principal efeito é reduzir a variância de modelos instáveis, e o treino pode ser paralelizado.

## “O que é out-of-bag?”

> Para cada modelo, são os registros que não entraram em sua amostra bootstrap. Cada registro pode ser previsto pelas unidades que não o viram, e a agregação dessas previsões fornece uma estimativa OOB de generalização. Ela é útil para validação interna, mas não substitui o teste final.

## “Como funciona a Random Forest?”

> É um Bagging de árvores que também seleciona aleatoriamente um subconjunto de atributos candidatos em cada corte. O bootstrap cria diversidade nos registros e `max_features` reduz a correlação entre árvores. A floresta agrega por voto ou média e tende a ter menor variância que uma árvore isolada.

## “Random Forest precisa de normalização?”

> Em geral, não. As árvores usam ordem e limiares por atributo, e não uma distância global. Ainda assim, categorias, ausências e requisitos da implementação precisam de tratamento compatível.

## “Mais árvores causam overfitting na Random Forest?”

> Em geral, aumentar apenas `n_estimators` estabiliza o ensemble e o erro tende a atingir um platô. O custo e a memória continuam crescendo. Overfitting depende mais da estrutura das árvores, dados, vazamento e ajuste de hiperparâmetros do que simplesmente de haver muitas árvores.

## “Como funciona o Boosting?”

> Boosting treina modelos em sequência, de modo que cada estágio corrige ou complementa o conjunto anterior. AdaBoost faz isso reponderando observações e estimadores; Gradient Boosting ajusta gradientes ou resíduos. A combinação final usa todos os estágios.

## “Como funciona o AdaBoost?”

> Inicialmente todas as observações têm peso `1/N`. A cada rodada, um estimador é ajustado, calcula-se seu erro ponderado e define-se seu peso de voto. Exemplos errados ganham influência relativa para a rodada seguinte. A predição final é uma soma ou votação ponderada pelos pesos dos estimadores.

## “Bagging ou Boosting?”

> Eu escolheria Bagging quando o problema principal for a instabilidade e alta variância do modelo-base, especialmente árvores. Consideraria Boosting quando precisar reduzir viés e construir uma função mais expressiva em etapas. A decisão final depende de validação, ruído, custo, latência e interpretabilidade.

## “Qual é o custo computacional?”

> Não há uma classe única como linear ou quadrática sem especificar o modelo-base. Com `B` estimadores, treino e predição crescem aproximadamente com `B`. Bagging permite paralelizar os modelos; Boosting é sequencial. Em Random Forest, uma aproximação de treino é `O(B·m·N log N)` em condições balanceadas, e a predição é aproximadamente `O(B·h)` por exemplo.

## “Ensemble é paramétrico?”

> Ensemble é uma estratégia, então a resposta depende de seus modelos-base. Random Forest e AdaBoost com árvores são normalmente chamados de não paramétricos, mas ainda aprendem estruturas e pesos e possuem hiperparâmetros. Não paramétrico não significa ausência de configuração ou de valores aprendidos.

---

# Parte XII — Comparação final

| Aspecto | Bagging | Random Forest | AdaBoost |
|---|---|---|---|
| Família | Agregação por reamostragem | Bagging de árvores com atributos aleatórios | Boosting adaptativo |
| Treino | Independente | Independente entre árvores | Sequencial |
| Modelo-base | Qualquer compatível | Árvore | Estimador que aceite pesos, frequentemente árvore rasa |
| Foco típico | Reduzir variância | Reduzir variância e correlação | Reduzir erro/viés adaptativamente |
| Agregação | Voto/média | Voto/média | Voto/soma ponderada |
| Escala | Depende da base | Geralmente desnecessária | Depende da base; com árvores, geralmente desnecessária |
| Outliers | Impacto depende da base | Não imune | Pode enfatizá-los fortemente |
| Desbalanceamento | Não resolve sozinho | Não resolve sozinho | Não resolve sozinho |
| Paralelização | Alta | Alta | Limitada entre estágios |
| Validação interna | OOB possível | OOB possível | Não é OOB por padrão |
| Custo | Multiplica custo da base | Muitas árvores e muitos nós | Muitos estágios sequenciais |

---

# Parte XIII — Roteiro de treinamento

## Sessão 1 — Fundamentos e viés–variância

- definir ensemble, diversidade e agregação;
- diferenciar viés, variância, overfitting e underfitting;
- interpretar gráfico de complexidade e alvo de tiros.

Critério de domínio: explicar por que vários modelos podem ser melhores e quando não serão.

## Sessão 2 — Bagging e bootstrap

- simular duas amostras bootstrap;
- calcular a aproximação `63,2%/36,8%`;
- explicar voto, média e OOB;
- discutir escala, outliers e desbalanceamento.

Critério de domínio: descrever treino, predição e avaliação OOB sem confundir reposição.

## Sessão 3 — Random Forest

- explicar a dupla aleatoriedade de linhas e atributos;
- relacionar `max_features` a força e correlação;
- revisar critérios, hiperparâmetros e custo;
- comparar uma árvore com uma floresta.

Critério de domínio: defender por que Random Forest reduz variância e por que não precisa de scaler.

## Sessão 4 — Boosting e AdaBoost

- simular pesos iniciais `1/N`;
- calcular erro ponderado e `α`;
- atualizar pesos de exemplos certos e errados;
- calcular o voto final ponderado.

Critério de domínio: explicar por que o último estimador não é a predição final e por que ruído é perigoso.

## Sessão 5 — Seleção prática

- comparar Bagging, Random Forest e Boosting;
- escolher métricas para desbalanceamento;
- propor validação sem vazamento;
- justificar custo, memória, latência e interpretabilidade.

Critério de domínio: recomendar um método e também dizer em quais condições mudaria a recomendação.

---

# Checklist final de domínio

Você domina o conteúdo quando consegue, sem consultar material:

- definir ensemble e explicar a importância da diversidade;
- diferenciar voting, Bagging, Random Forest, Boosting e Stacking;
- explicar viés, variância, overfitting e underfitting corretamente;
- explicar bootstrap com reposição e a aproximação `63,2%/36,8%`;
- descrever OOB e suas limitações;
- justificar por que Bagging reduz variância;
- explicar a dupla aleatoriedade da Random Forest;
- distinguir parâmetros aprendidos de hiperparâmetros;
- prever os efeitos de `n_estimators`, `max_features`, profundidade e tamanho de folha;
- explicar por que escala depende do modelo-base;
- discutir outliers e desbalanceamento sem afirmar imunidade;
- calcular pesos iniciais, erro, `α` e atualização do AdaBoost;
- explicar voto ponderado no AdaBoost;
- comparar custos de treino, predição, memória e paralelização;
- detectar todas as pegadinhas listadas acima;
- propor um pipeline de validação sem vazamento e alinhado ao custo do erro.
