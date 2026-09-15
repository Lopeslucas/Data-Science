# DataPrep

---
# Pontos cobertos pelo questionario abaixo:
- Tecnicas de identificação de outliers ✅


---

## Por que o KNN sofre com a chamada maldição da dimensionalidade?
O KNN sofre com a maldição da dimensionalidade porque, conforme aumentamos o número de features, o espaço fica mais esparso e as distâncias entre as observações tendem a ficar cada vez mais semelhantes. Com isso, fica mais difícil distinguir quem é realmente um vizinho próximo, prejudicando a capacidade preditiva do KNN. Para mitigar, podemos fazer seleção de variáveis, redução de dimensionalidade e, quando possível, aumentar a quantidade de dados.

Um segundo efeito é que, para preencher adequadamente um espaço de muitas dimensões, precisamos de muito mais dados. Com poucos dados, a vizinhança fica muito esparsa


## Qual é a diferença entre distância Euclidiana, Manhattan e Minkowski no KNN?
A distância Euclidiana mede a distância em linha reta e é um caso da Minkowski com p = 2. A Manhattan soma as diferenças absolutas entre os atributos e corresponde à Minkowski com p = 1. A Minkowski é uma generalização das duas e permite controlar a métrica pelo parâmetro p.

## Por que o KNN sofre com a maldição da dimensionalidade? O que acontece com as distâncias quando o número de features cresce muito e como você poderia mitigar esse problema?
O KNN sofre com a maldição da dimensionalidade porque, à medida que o número de features aumenta, o espaço se torna mais esparso e as distâncias entre as observações tendem a ficar mais semelhantes. Com isso, a noção de vizinho próximo perde poder discriminativo, prejudicando a predição. Podemos mitigar o problema com seleção de features, redução de dimensionalidade como PCA e, quando possível, maior quantidade de dados. Também é importante fazer scaling, porque o KNN depende diretamente das distâncias

## No KNN, imagine dois cenários:
- K = 1
- K = 50
Como o aumento de K afeta viés, variância e risco de overfitting/underfitting?


Com K igual a 1, o KNN fica muito sensível aos dados de treino, apresentando baixo viés, alta variância e maior risco de overfitting. À medida que K aumenta, a fronteira de decisão fica mais suave, o viés aumenta, a variância diminui e, se K for muito alto, o modelo pode sofrer underfitting.”
Frase para guardar:

- K pequeno \Rightarrow baixo\ viés,\ alta\ variância
- K grande \Rightarrow alto\ viés,\ baixa\ variância
