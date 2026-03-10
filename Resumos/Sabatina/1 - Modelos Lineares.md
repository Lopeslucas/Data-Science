# Modelos Lineares

O conceito dos métodos lineares, como a regressão linear simples e múltipla, baseia-se na premissa de que existe uma relação aproximadamente em linha reta entre as variáveis preditoras (X) e a variável de resposta (Y),. Matematicamente, esses métodos modelam a resposta como uma combinação linear de coeficientes e preditores, servindo como uma ferramenta fundamental para predição e inferência estatística.

Premissas Fundamentais
Para que os modelos lineares sejam válidos e eficazes, assume-se que:
- Linearidade: A relação entre os preditores e a resposta é retilínea, o que significa que a mudança na resposta por unidade de variação no preditor é constante.
- Aditividade: O efeito de um preditor na resposta é independente dos valores dos outros preditores no modelo.
- Independência e Média Zero dos Erros: Os termos de erro são independentes entre si e, em média, somam zero.
- Homocedasticidade: Os termos de erro possuem uma variância constante, não variando sistematicamente com os valores dos preditores ou da resposta.

Vantagens
- Interpretabilidade: São modelos extremamente fáceis de explicar e entender, permitindo identificar claramente quais variáveis estão associadas à resposta e a força dessa associação.
- Ferramentas de Inferência: Oferecem uma base sólida para testes de hipóteses, permitindo o cálculo de erros padrão, intervalos de confiança e p-values para validar as descobertas.
- Eficiência e Simplicidade: São computacionalmente baratos para ajustar e servem como um excelente ponto de partida antes de tentar métodos mais complexos.
- Competitividade: Em muitos problemas do mundo real, a aproximação linear é surpreendentemente competitiva em relação a métodos não lineares sofisticados.

Desvantagens
- Rigidez (Alto Viés): A suposição de linearidade é frequentemente uma simplificação excessiva da realidade; se a relação real for complexa ou curva, o modelo terá um viés alto e baixa precisão,.
- Problemas em Alta Dimensionalidade: Quando o número de preditores (p) é grande em relação ao número de observações (n), o modelo pode sofrer de overfitting (sobreajuste); se p>n, o método de mínimos quadrados nem sequer possui uma solução única.
- Sensibilidade a Dados Atípicos: O ajuste pode ser drasticamente distorcido por outliers (valores incomuns na resposta) ou pontos de alta alavancagem (valores incomuns nos preditores).
- Colinearidade: Quando os preditores são altamente correlacionados entre si, torna-se difícil separar os efeitos individuais de cada variável, aumentando a incerteza das estimativas.

---
# Perguntas e respostas:

## Em regressão linear, quais são as principais premissas do modelo e o que acontece se elas forem violadas?
A regressão linear assume algumas premissas importantes.

    a. Linearidade, ou seja, a relação entre as variáveis independentes e a variável alvo deve ser aproximadamente linear nos parâmetros.
    b. Independência dos erros, ou seja, os resíduos não devem estar correlacionados entre si.
    c. Homocedasticidade, que significa que a variância dos erros deve ser constante ao longo dos valores previstos.
    d. Normalidade dos resíduos, que é importante principalmente para inferência estatística.
    e. Ausência de multicolinearidade forte entre as variáveis explicativas, pois isso pode tornar os coeficientes instáveis.
    f. Quando essas premissas são violadas, podemos ter problemas como coeficientes não confiáveis, pior generalização ou inferências incorretas.

## Lasso x Ridge
L1 e L2 são técnicas de regularização usadas para evitar overfitting penalizando os coeficientes do modelo.

Na regularização L1, também chamada de Lasso, a penalização é feita pela soma dos valores absolutos dos coeficientes. Isso pode fazer com que alguns coeficientes se tornem exatamente zero, o que permite seleção automática de variáveis e gera modelos mais esparsos.

Na regularização L2, chamada Ridge, a penalização é feita pela soma dos quadrados dos coeficientes. Nesse caso os coeficientes são reduzidos, mas dificilmente zerados, o que ajuda a lidar melhor com multicolinearidade e melhora a estabilidade do modelo.

Em geral usamos L1 quando queremos seleção de variáveis ou temos alta dimensionalidade, e L2 quando queremos reduzir overfitting e lidar com variáveis correlacionadas.


## Qual a diferença entre regressão linear e regressão logística, e por que não podemos usar regressão linear para classificação?
A regressão linear modela uma relação linear entre as variáveis independentes e a variável alvo, estimando valores contínuos e minimizando o erro quadrático médio.

Já a regressão logística é usada para classificação e modela a probabilidade da classe usando a função logística, ou sigmoide, garantindo que a saída fique entre 0 e 1.

Não usamos regressão linear para classificação porque ela pode prever valores menores que 0 ou maiores que 1, o que não faz sentido para probabilidades. Além disso, a função de custo da regressão linear não é adequada para problemas de classificação.

A regressão logística utiliza máxima verossimilhança e log loss, o que torna o modelo mais adequado para estimar probabilidades e tomar decisões de classificação.