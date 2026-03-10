## Por que árvore de decisão não precisa de normalização, mas regressão precisa?
Modelos de árvore de decisão não precisam de normalização porque eles fazem divisões baseadas em regras do tipo feature menor que um determinado valor. Essas divisões dependem apenas da ordenação dos valores e não da escala, então não importa se uma variável está entre 0 e 100 ou entre 0 e 10000.

Já modelos lineares e modelos baseados em otimização, como regressão, SVM e KNN, são sensíveis à escala porque os coeficientes e o processo de treinamento dependem da magnitude das variáveis. Se as variáveis estiverem em escalas muito diferentes, o modelo pode dar mais peso para variáveis com valores maiores, além de dificultar a convergência do algoritmo.

Por isso, nesses modelos é comum usar normalização ou padronização, enquanto em árvores isso não é necessário.


## Qual a diferença entre Random Forest e Gradient Boosting?
Random Forest é um método de ensemble baseado em Bagging, onde várias árvores são treinadas de forma independente usando bootstrap sampling dos dados.

Além disso, em cada split é usado um subconjunto aleatório de features, o que reduz a correlação entre as árvores e diminui a variância do modelo. A previsão final é feita pela média ou voto das árvores.

Já o Gradient Boosting é um método sequencial, onde cada nova árvore é treinada para corrigir os erros da anterior, otimizando gradualmente a função de perda através do gradiente.
Isso reduz o viés e permite aprender padrões complexos, mas torna o modelo mais sensível a overfitting e aos hiperparâmetros.

Em geral, Random Forest é mais robusto e fácil de usar, enquanto Gradient Boosting costuma ter melhor performance, mas exige mais tuning.


## Em quais situações um modelo linear pode performar melhor que uma árvore de decisão?
Modelos lineares podem performar melhor que árvores quando a relação entre as variáveis é aproximadamente linear, porque nesse caso o modelo consegue capturar o padrão sem precisar de muita complexidade.

Eles também costumam funcionar melhor em datasets menores, porque têm menor variância e menos risco de overfitting, enquanto árvores de decisão tendem a crescer muito e se ajustar demais aos dados de treino.

Outra vantagem dos modelos lineares é a interpretabilidade, já que os coeficientes mostram diretamente a influência de cada variável.

Além disso, modelos lineares costumam ter menor custo computacional e funcionam bem quando os dados não têm muita não-linearidade ou ruído.

Já árvores são mais indicadas quando existem interações complexas entre variáveis ou relações não lineares.


## Qual a diferença de viés e variância entre modelos lineares, árvores de decisão e Random Forest?
Modelos lineares costumam ter alto viés e baixa variância, porque eles assumem uma relação linear entre as variáveis, então são mais simples e menos sensíveis a variações nos dados, o que reduz overfitting, mas pode causar underfitting quando os dados são mais complexos.

Árvores de decisão têm baixo viés e alta variância, porque conseguem se ajustar muito bem aos dados de treino, capturando padrões complexos, mas pequenas mudanças nos dados podem gerar árvores muito diferentes, o que aumenta o risco de overfitting.

Já o Random Forest reduz a variância da árvore de decisão usando bagging, com bootstrap e várias árvores independentes, e depois combinando as previsões pela média ou voto.

Com isso, o Random Forest fica com variância menor que a árvore simples e um viés um pouco maior, mas com melhor capacidade de generalização.

