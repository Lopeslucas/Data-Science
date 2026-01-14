# Análise e Predição de Indenização em Seguro Auto (Base SUSEP)

## Contexto de Negócio
No mercado de Seguros Auto, especialmente no contexto corporativo (PJ), a correta avaliação de risco de indenização é fundamental para:
•	Reduzir custos com sinistros
•	Priorizar análises manuais
•	Apoiar decisões de precificação e aceitação de risco
•	Melhorar a eficiência operacional

Neste cenário, a área de dados atua como suporte à tomada de decisão, oferecendo modelos preditivos e análises descritivas que ajudam a antecipar eventos de indenização.

Este projeto utiliza uma base pública da SUSEP (Superintendência de Seguros Privados) para simular um problema real enfrentado por seguradoras.
---

## Objetivo do Projeto
Desenvolver uma análise exploratória e um modelo de Machine Learning capazes de estimar a probabilidade de ocorrência de indenização em apólices de seguro auto, apoiando decisões de negócio relacionadas à gestão de risco.

O modelo não substitui decisões humanas, mas atua como um mecanismo de priorização, indicando apólices com maior risco relativo.
---

## Perguntas de Negócio
O projeto busca responder às seguintes perguntas:
1.	Quais características das apólices e dos segurados estão mais associadas à ocorrência de indenização?
2.	É possível identificar perfis de maior risco com base em variáveis demográficas, regionais e contratuais?
3.	Como um modelo preditivo pode auxiliar na priorização de análises manuais?
4.	Qual seria o impacto potencial da utilização desse modelo no processo operacional de uma seguradora?
---

## Definição do Problema Analítico
O problema foi estruturado como uma tarefa de classificação binária, onde o objetivo é prever se uma apólice resultará ou não em indenização.

Variável Alvo

FLAG_INDENIZ_POS
•	1 → Apólice com indenização paga (evento positivo)
•	0 → Apólice sem indenização

A escolha de uma variável binária permite:
•	Simplicidade de interpretação
•	Uso de modelos interpretáveis (ex: Regressão Logística)
•	Alinhamento com decisões operacionais (analisar vs não analisar)
---

## Base de Dados
•	Fonte: SUSEP – Dados públicos de seguros
•	Tipo: Dados estruturados
•	Granularidade: Apólice / Sinistro
---

##  Análise Exploratória (EDA)
Durante a etapa de Data Understanding, foram realizadas:
•	Análise de cardinalidade e dominância
•	Tratamento de datas inválidas (00000000)
•	Identificação de inconsistências temporais
•	Criação de variáveis derivadas (Feature Engineering)
•	Análise de balanceamento da variável alvo

Essas análises permitiram:
•	Melhor compreensão do comportamento dos sinistros
•	Identificação de possíveis vieses e problemas nos dados
•	Criação de variáveis mais informativas para o modelo
---

## Feature Engineering
Foram criadas variáveis para enriquecer o modelo, como:
•	Consistência temporal entre eventos
•	Quantidade de dias até aviso e liquidação
•	Identificação de duplicidade de sinistros
•	Idade limpa do beneficiário
•	Faixas etárias para melhor interpretação atuarial

Essas transformações tiveram como foco aumentar o valor informacional dos dados, mantendo coerência com o negócio.
---

## Modelagem
O primeiro modelo implementado foi a Regressão Logística, escolhida por:
•	Alta interpretabilidade
•	Facilidade de explicação para áreas de negócio
•	Uso frequente em contextos regulados como seguros

A avaliação do modelo foi realizada utilizando métricas adequadas ao problema de classificação, considerando o impacto de falsos positivos e falsos negativos.
---

## Métricas e Avaliação
As métricas foram analisadas com foco em:
•	Capacidade de discriminação do modelo
•	Trade-off entre capturar sinistros relevantes e custo operacional
•	Alinhamento entre métrica técnica e impacto de negócio

Em um cenário de seguros, deixar de identificar uma apólice de alto risco tende a ser mais custoso do que analisar um falso positivo.
---

## Simulação de Impacto (Conceitual)
Mesmo sem dados financeiros reais, o modelo permite simular cenários como:
•	Priorização de X% das apólices com maior risco
•	Concentração de grande parte dos sinistros esperados em um subconjunto menor de contratos
•	Redução de esforço operacional com foco em risco

Essas simulações ajudam a conectar o modelo à mensuração de valor para o negócio.
---

## Ciclo de Vida do Modelo (CRISP-DM)
O projeto segue as etapas do CRISP-DM:
1.	Business Understanding
2.	Data Understanding
3.	Data Preparation
4.	Modeling
5.	Evaluation
6.	Deployment (conceitual)
---

## Conclusão
Este projeto demonstra como Ciência de Dados aplicada a Seguros pode apoiar decisões estratégicas, combinando:
•	Análise exploratória orientada a negócio
•	Modelagem interpretável
•	Foco em impacto e valor gerado

** Este projeto tem caráter educacional e demonstrativo, utilizando dados públicos para simular um problema real de mercado.**
