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