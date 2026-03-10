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