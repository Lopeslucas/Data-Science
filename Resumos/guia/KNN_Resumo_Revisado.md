# KNN — resumo revisado para sabatina

## 1. Visão geral

O **K-Nearest Neighbors (KNN)** é um algoritmo supervisionado que prediz uma observação com base nos `K` exemplos de treinamento mais próximos.

- **Classificação:** usa votação entre os vizinhos.
- **Regressão:** usa, normalmente, a média ou média ponderada dos valores dos vizinhos.
- É **não paramétrico**: não pressupõe uma forma fixa para a relação entre entrada e saída.
- É baseado em **instâncias** e possui aprendizado tardio (*lazy learning*): o treinamento consiste principalmente em armazenar os dados.

> Pegadinha: KNN é preditivo e supervisionado. Ele não é um algoritmo de agrupamento apenas porque usa proximidade.

## 2. Como funciona

Para uma nova observação `x`:

1. Calcula-se a distância de `x` para os registros de treino.
2. Selecionam-se os `K` menores valores.
3. Agregam-se as respostas dos vizinhos.

Classificação com pesos uniformes:

```text
classe(x) = classe mais frequente entre os K vizinhos
```

Classificação ponderada pela distância:

```text
peso_i = 1 / (distância_i + ε)

classe(x) = classe com maior soma de pesos
```

Regressão:

```text
ŷ = média(y dos K vizinhos)
```

O termo `ε` evita divisão por zero. Implementações podem tratar distância zero de modo específico.

## 3. O papel de K

`K` é o número de vizinhos considerados e um dos hiperparâmetros mais importantes.

### K pequeno

- fronteira mais irregular;
- baixo viés e alta variância;
- maior sensibilidade a ruído e outliers;
- maior risco de overfitting;
- `K = 1` quase memoriza o conjunto de treino.

### K grande

- fronteira mais suave;
- maior viés e menor variância;
- risco de underfitting;
- maior influência da classe majoritária;
- se `K = N`, a classificação uniforme tende sempre à classe mais frequente.

Para classificação binária com pesos uniformes, um `K` ímpar reduz empates, mas não os impossibilita em problemas multiclasse.

**Boa prática:** escolher `K` por validação cruzada dentro de um pipeline de pré-processamento.

## 4. Métricas de distância

### Euclidiana

```text
d(x, y) = sqrt(Σ (x_j - y_j)²)
```

É a distância em linha reta. Corresponde à Minkowski com `p = 2`.

### Manhattan

```text
d(x, y) = Σ |x_j - y_j|
```

Corresponde à Minkowski com `p = 1` e pode ser menos sensível a diferenças extremas por atributo.

### Minkowski

```text
d(x, y) = (Σ |x_j - y_j|^p)^(1/p)
```

### Hamming

Conta posições diferentes. É apropriada para vetores binários ou categorias adequadamente representadas.

### Mahalanobis

```text
d(x, y) = sqrt((x-y)ᵀ S⁻¹ (x-y))
```

Considera escala e correlação entre variáveis, mas exige estimar e inverter uma matriz de covariância, o que aumenta o custo e pode ser instável.

> A métrica deve representar a noção de similaridade do problema. Não existe uma distância universalmente melhor.

## 5. Escala e preparação dos dados

Como a decisão depende diretamente das distâncias, variáveis de maior escala podem dominar o resultado.

Exemplo: uma variável entre `0 e 100000` tende a pesar mais que outra entre `0 e 1`.

Alternativas comuns:

- **StandardScaler:** média 0 e desvio-padrão 1;
- **MinMaxScaler:** transforma para um intervalo, frequentemente `[0, 1]`;
- **RobustScaler:** usa mediana e intervalo interquartil, sendo mais robusto a extremos;
- **MaxAbsScaler:** útil especialmente para dados esparsos.

O scaler deve ser ajustado **somente no treino** e aplicado à validação/teste.

Para variáveis categóricas, a codificação deve preservar uma distância coerente. Atribuir números arbitrários a categorias nominais cria uma ordem artificial.

## 6. Outliers, ruído e desbalanceamento

### Outliers

Podem alterar vizinhanças, especialmente com `K` pequeno. Antes de remover:

1. verifique erro de coleta;
2. avalie se é um evento raro legítimo;
3. compare o desempenho com tratamento dentro da validação.

Possíveis respostas: `K` maior, pesos por distância, scaler robusto, seleção de variáveis e tratamento justificado dos extremos.

### Classes desbalanceadas

A votação pode favorecer a classe majoritária. Pesos por distância ajudam apenas quando a minoria forma regiões locais coerentes; não resolvem todo desbalanceamento.

Também podem ser usados:

- reamostragem apenas no treino;
- métricas como recall, precisão, F1 e PR-AUC;
- escolha de `K` e da métrica por validação estratificada;
- métodos que ponderem classes ou amostras, quando disponíveis.

## 7. Dimensionalidade

Em muitas dimensões, as distâncias tendem a ficar menos discriminativas: é a **maldição da dimensionalidade**.

Consequências:

- vizinhos “próximos” deixam de ser realmente próximos;
- o conjunto necessário cresce rapidamente;
- KD Tree e Ball Tree perdem eficiência;
- variáveis irrelevantes adicionam ruído à distância.

Mitigações: seleção de variáveis, PCA quando aceitável, engenharia de atributos e escolha cuidadosa da métrica.

## 8. Parâmetros aprendidos e hiperparâmetros

KNN praticamente não aprende coeficientes. Seu conhecimento está nas instâncias armazenadas.

Hiperparâmetros típicos:

- `n_neighbors`: valor de `K`;
- `weights`: `uniform`, `distance` ou função customizada;
- `metric`: métrica de distância;
- `p`: potência da Minkowski;
- `algorithm`: `brute`, `kd_tree`, `ball_tree` ou `auto`;
- `leaf_size`: tamanho das folhas das estruturas de busca.

> `K` não é parâmetro aprendido; é hiperparâmetro escolhido antes do ajuste e validado nos dados.

## 9. Custo computacional

Se `N` é o número de exemplos e `P` o de variáveis:

- treino simples: aproximadamente `O(NP)` para armazenar/copiar os dados;
- memória: `O(NP)`;
- previsão por consulta com força bruta: `O(NP)` para calcular distâncias, além da seleção dos vizinhos;
- previsão em `M` consultas: aproximadamente `O(MNP)`.

O custo está principalmente na **predição**, não no treino.

### Estratégias de busca

- **Brute force:** robusta e frequentemente competitiva em alta dimensão ou bases menores.
- **KD Tree:** particiona o espaço por eixos; costuma funcionar melhor em baixa dimensão.
- **Ball Tree:** particiona em regiões aproximadamente esféricas e pode ajudar com outras métricas.

Nenhuma estrutura garante busca logarítmica em todos os cenários. Dimensão, métrica, tamanho da base e distribuição dos dados importam.

## 10. Pontos positivos

- conceito simples;
- praticamente nenhum custo de ajuste;
- modela fronteiras não lineares;
- atende classificação e regressão;
- naturalmente multiclasse;
- fácil de atualizar adicionando novas instâncias.

## 11. Pontos negativos e como reduzir

| Limitação | Como lidar |
|---|---|
| Predição cara | KD/Ball Tree quando adequados, redução dimensional ou busca aproximada |
| Alto uso de memória | seleção de amostras/variáveis e compressão adequada |
| Sensível à escala | scaler dentro de pipeline |
| Sensível a ruído/outliers | validar `K`, pesos por distância e tratamento robusto |
| Sofre em alta dimensão | seleção de variáveis ou redução dimensional |
| Desbalanceamento | validação estratificada, métricas adequadas e reamostragem |
| Baixa explicabilidade global | explicar vizinhos e distâncias localmente |

## 12. Pegadinhas de sabatina

- KNN é **supervisionado, preditivo, não paramétrico e baseado em instâncias**.
- Classifica e também faz regressão.
- Normalização não muda só velocidade; muda a geometria e pode mudar a predição.
- `weights="distance"` não escolhe automaticamente o melhor `K`.
- `K` grande não torna o modelo mais complexo; tende a suavizá-lo.
- `K = N` com votação uniforme tende à classe majoritária.
- KD Tree não é automaticamente melhor que força bruta.
- Distância de Hamming é útil para atributos binários/categóricos compatíveis.
- Empates podem depender da regra da biblioteca e da ordem/representação das classes.

## 13. Resposta curta para entrevista

> KNN é um algoritmo supervisionado e não paramétrico que usa a resposta dos K exemplos mais próximos. Ele quase não treina, mas transfere o custo para a predição. Seus pontos críticos são escolher K e a métrica, padronizar as variáveis, controlar ruído, dimensionalidade e desbalanceamento. K pequeno tende a alta variância; K grande tende a alto viés.

## 14. Checklist

- [ ] A métrica representa similaridade no domínio?
- [ ] O pré-processamento está dentro de um pipeline?
- [ ] O scaler foi ajustado somente no treino?
- [ ] `K`, pesos e métrica foram validados?
- [ ] Há variáveis irrelevantes ou dimensão excessiva?
- [ ] O conjunto está desbalanceado?
- [ ] O custo de inferência atende ao ambiente de produção?
- [ ] A métrica de avaliação corresponde ao custo do erro?
