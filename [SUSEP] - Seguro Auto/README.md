# Análise e Predição de Indenização em Seguro Auto (Base SUSEP)

## Contexto de Negócio
No mercado de Seguros Auto, especialmente no contexto corporativo (PJ), a correta avaliação de risco de indenização é fundamental para:
- Reduzir custos com sinistros
- Priorizar análises manuais
- Apoiar decisões de precificação e aceitação de risco
- Melhorar a eficiência operacional

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
- 1 → Apólice com indenização paga (evento positivo)
- 0 → Apólice sem indenização

A escolha de uma variável binária permite:
- Simplicidade de interpretação
- Uso de modelos interpretáveis (ex: Regressão Logística)
- Alinhamento com decisões operacionais (analisar vs não analisar)
---

## Base de Dados
- Fonte: SUSEP – Dados públicos de seguros
- Tipo: Dados estruturados
- Granularidade: Apólice / Sinistro
---

##  Análise Exploratória (EDA)
Durante a etapa de Data Understanding, foram realizadas:

### Técnicas Aplicadas
1. **Análise de Cardinalidade e Dominância**
   - Mapeamento da distribuição de classes em variáveis categóricas (MODALIDADE, TIPO_PROD, COBERTURA, EVENTO, CAUSA)
   - Identificação de classes dominantes e raras para decisão de agrupamento

2. **Limpeza e Normalização de Datas**
   - Tratamento de preenchimentos inválidos (00000000, valores fora do domínio)
   - Validação de data de nascimento (ano ≥ 1900)
   - Validação de datas de ocorrência (ano ≥ 1990)
   - Conversão para formato datetime com tratamento de erros

3. **Validação de Consistência Temporal**
   - Verificação de relação lógica: D_OCORRÊNCIA ≤ D_AVISO ≤ D_LIQUIDAÇÃO
   - Identificação de inconsistências (registros onde ocorrência é posterior ao aviso)
   - Cálculo de indicadores temporais: dias até aviso e dias até liquidação

4. **Identificação de Duplicidade de Sinistros**
   - Criação de ID único de sinistro (COD_APO + ITEM + EVENTO + D_OCORR)
   - Detecção de múltiplos registros referentes ao mesmo evento
   - Análise de valores inconsistentes em sinistros duplicados

5. **Qualidade de Dados e Outliers**
   - Validação de idade (16 a 100 anos)
   - Análise de relações entre VAL_SALVAD, VAL_RESS e INDENIZ
   - Limpeza de 409.440 registros com idade fora dos limites

### Decisões de Negócio Resultantes
- **Agrupamento de MODALIDADE**: Classes 2 e 3 agrupadas como "outras" (representam <3% dos dados)
- **Agrupamento de COBERTURA**: Classes 9, 2 e 4 agrupadas como "outras" (representam <2% dos dados)
- **Agrupamento de EVENTO**: Eventos 3-7 agrupados como "outros" (representam <1% dos dados)
- **Agrupamento de CAUSA**: Causas 1-6 agrupadas como "outras" (representam <5% dos dados)
- **Tratamento de valores inválidos**: Código 0 em variáveis categóricas removido como ruído administrativo
- **Recorte de população**: Exclusão de registros com idade <16 ou >100 (rejustamento: 409.440 registros)
- **Definição da variável-alvo**: PAGOU_INDENIZACAO = 1 se INDENIZ > 0, else 0 (independente de salvados/ressarcimentos)
---

## Feature Engineering
Foram criadas variáveis para enriquecer o modelo, mantendo coerência com o negócio:

### Variáveis Derivadas Implementadas

1. **Variáveis Temporais**
   - `DIAS_ATE_AVISO`: Intervalo entre ocorrência e aviso do sinistro
   - `DIAS_ATE_LIQ`: Intervalo entre ocorrência e liquidação
   - **Justificativa negócio**: Sinistros avisados muito tempo após a ocorrência podem indicar fraude ou processos administrativos deficientes

2. **Flags de Consistência Temporal**
   - `FLAG_OCORR_MAIOR_AVI`: Identifica ocorrências posteriores ao aviso (incoerência lógica)
   - `FLAG_AVI_MAIOR_LIQ`: Identifica avisos posteriores à liquidação
   - **Justificativa negócio**: Permite identificar registros com qualidade duvidosa ou regras de negócio não respeitadas

3. **Identificação de Sinistros Múltiplos**
   - `ID_SINISTRO`: Identificador único (COD_APO + ITEM + EVENTO + D_OCORR)
   - `FLAG_DUPLICIDADE_ID`: Indica múltiplas linhas do mesmo sinistro
   - **Justificativa negócio**: Base SUSEP pode registrar o mesmo sinistro em múltiplas linhas (ex: múltiplas coberturas, ajustes posteriores)

4. **Variáveis de Indenização**
   - `PAGOU_INDENIZACAO`: Variável-alvo binária (1 se INDENIZ > 0)
   - `FLAG_SALVADO_MAIOR_IDENIZ`: Identifica salvados maiores que indenização
   - **Justificativa negócio**: Alguns sinistros podem ter salvados recuperados maiores que o valor pago, indicando ajustes contábeis

5. **Variáveis Demográficas**
   - `IDADE`: Idade do condutor sinistrado na data da ocorrência
   - `FAIXA_IDADE`: Agrupamento em 8 categorias (18-20, 21-25, 26-30, 31-40, 41-50, 51-60, 61-70, 71+)
   - **Justificativa negócio**: Idade é proxy para experiência como condutor; faixas etárias permitem tarifação por perfil de risco

### Decisões de Feature Engineering
- **Idade limpa**: Filtro de domínio (16-100 anos) para remover registros administrativamente inválidos
- **Faixas etárias**: Distribuição não uniforme deliberada, alinhada com categorias atuariais do mercado de seguros
- **Variáveis temporais em dias**: Escolha de escala linear para melhor interpretação de impacto negócio
---

## Modelagem

### Pipeline de Pré-processamento
A transformação dos dados foi implementada através de um `ColumnTransformer` com estratégias diferenciadas por tipo de variável:

**Variáveis Numéricas** (2 features)
- `StandardScaler`: Padronização para média 0 e desvio 1
- Variáveis: IDADE, ANO_MODELO
- Justificativa: Regressão Logística é sensível a escala; padronização melhora convergência
- **Nota**: DIAS_ATE_AVISO foi excluída para evitar data leakage (usa D_AVI, data futura)

**Pré-processamento de Tipo de Dados**
- Conversão de TODAS as variáveis categóricas para `str` antes do pipeline
- Justificativa: Evita erro de tipo misto (int/str) no OneHotEncoder durante fit do modelo
- Implementado: `df_treated[col] = df_treated[col].astype(str)` para cada coluna categórica

**Variáveis Categóricas de Baixa Cardinalidade** (7 features)
- `OneHotEncoder`: Codificação one-hot (drop='first' para evitar multicolinearidade)
- Variáveis: SEXO, MODALIDADE, TIPO_PROD, COBERTURA, EVENTO, CAUSA, REGIAO
- Justificativa: Categorias com poucos valores únicos; one-hot permite que modelo capture efeitos não-lineares

**Variáveis Categóricas de Alta Cardinalidade** (2 features)
- `TargetEncoder` (smoothing=5): Codificação baseada em valor da variável-alvo
- Variáveis: COD_MODELO, COD_TARIF
- Justificativa: Elevada cardinalidade (~14.000 modelos); target encoding reduz dimensionalidade mantendo informação preditiva, com smoothing para evitar overfitting

### Algoritmo de Classificação

**Regressão Logística**
- Classe: `LogisticRegression` (sklearn)
- Hiperparâmetros:
  - `max_iter=1000`: Iterações para convergência
  - `class_weight='balanced'`: Ajuste automático de pesos para desbalanceamento de classes
  - `n_jobs=-1`: Processamento paralelo
- Justificativa negócio: 
  - Modelo interpretável (coeficientes indicam direção e magnitude do efeito)
  - Probabilidades bem calibradas (importante para decisão de priorização operacional)
  - Baselinequick que permite iterações rápidas
  - Menos propenso a overfitting que modelos complexos

### Divisão Treino/Teste com Validação Temporal

**Split Temporal (não aleatório)**
- Cutoff: 2020-08-01
- Treino: D_AVI < 2020-08-01
- Teste: D_AVI ≥ 2020-08-01
- Justificativa negócio:
  - Dados são temporais (dinâmica de sinistros muda ao longo do tempo)
  - Simula cenário real de operação (treino com dados históricos, predição prospectiva)
  - Detecta dataset drift (mudanças nas distribuições que afetam modelo em produção)
  - Evita data leakage que distorceria performance relativa

Estamos assumindo que o modelo roda depois que o sinistro foi avisado, mas antes da análise/liquidação:
- OCORRÊNCIA → AVISO → **MODELO** → ANÁLISE → LIQUIDAÇÃO

---

## Métricas e Avaliação
As métricas foram selecionadas com foco em impacto de negócio:

### Métricas Implementadas

1. **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**
   - Métrica primária de discriminação do modelo
   - Independente do threshold de corte (importante para avaliação holística)
   - Justificativa negócio: Permite comparação com baselines; reflete capacidade do modelo discriminar risco alto vs baixo

2. **Classification Report** (Precision, Recall, F1-Score)
   - Avaliação por classe (indenizado vs não-indenizado)
   - Justificativa negócio: 
     - **Recall alto**: Identificar máximo de sinistros de alto risco (minimiza custos com sinistros perdidos)
     - **Precision alto**: Reduzir análises manuais desnecessárias (minimiza custo operacional)
     - **F1-Score**: Balanço entre ambos
4. **Cross-Validation (StratifiedKFold)**
   - Validação com 5 folds estratificados
   - **Importante**: Utiliza `pipe` (pipeline completo) ao invés do modelo isolado
   - Garante que preprocessamento seja aplicado corretamente em cada fold
   - Evita data leakage durante validação cruzada
   - Resultado: AUC = 0.8472 (±0.0011)


3. **ROC Curve (Visualization)**
   - Visualização trade-off entre TPR (True Positive Rate) e FPR (False Positive Rate)
   - Justificativa negócio: Permite escolher threshold operacional baseado em restrições de recursos humanos

### Trade-offs de Negócio

Em um cenário de seguros, o custo de erro varia:
- **Falso Negativo (caro)**: Deixar de identificar apólice de alto risco → sinistro maior que previsto
- **Falso Positivo (menos caro)**: Análise manual desnecessária → overhead operacional controlável

Logo, o modelo deve **priorizar Recall** sobre Precision, com threshold otimizado para capturar máximo de sinistros relevantes dentro da capacidade operacional disponível.

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

## Resumo de Técnicas e Decisões Aplicadas

### Técnicas de EDA
| Técnica | Aplicação | Resultado | Implementadas
| Feature | Tipo | Negócio | Status |
|---------|------|---------|--------|
| DIAS_ATE_AVISO | Intervalo temporal | ~~Proxy para velocidade de notificação~~ | ❌ Removida (data leakage) |
| DIAS_ATE_LIQ | Intervalo temporal | Proxy para complexidade/litígio | ✅ Mantida |
| FAIXA_IDADE | Categórica agrupada | Segmentação atuarial padrão | ✅ Implementada |
| ID_SINISTRO + FLAG_DUPLICIDADE | Booleana | Controle de qualidade e tratamento de agregação | ✅ Implementada
### Técnicas de Feature Engineering
| Feature | Tipo | Negócio |
|---------|------|---------|
| DIAS_ATE_AVISO | Intervalo temporal | Proxy para velocidade de notificação (indicador de fraude) |
| DIAS_ATE_LIQ | Intervalo temporal | Proxy para complexidade/litígio |
| FAIXA_IDADE | Categórica agrupada | Segmentação atuarial padrão |
| ID_SINISTRO + FLAG_DUPLICIDADE | Booleana | Controle de qualidade e tratamento de agregação |

### Técnicas de Modelagem
| Componente | Escolha | Justificativa |
|-----------|---------|----------------|
| Pré-processamento Numérico | StandardScaler | Convergência de regressão logística |
| Pré-processamento Categórico Baixo Card. | OneHotEncoder | Captura de efeitos não-lineares |
| Pré-processamento Categórico Alto Card. | TargetEncoder | Redução de dimensionalidade com informação preditiva |
| Algoritmo | Regressão Logística | Interpretabilidade + probabilidades calibradas |
| Split | Temporal (cutoff 2020-08-01) | Detecção de drift; simula cenário real |
| Balanceamento | class_weight='balanced' | Ajusta custo de erro entre classes |

##Exclusão de DIAS_ATE_AVISO | Remove variável baseada em D_AVI (informação futura não disponível no momento da predição) |
| Conversão de categorias para string | Garante tipos uniformes no preprocessamento; evita erro no OneHotEncoder |
| Uso de pipe em cross-validation | Mantém consistência: preprocessamento aplicado em cada fold |
| # Decisões de Negócio Aplicadas
| Decisão | Impacto |
|---------|---------|
| Variável-alvo = INDENIZ > 0 | Define evento de interesse: "pagamento realizado" |
| Recorte de idade 16-100 | Limpeza de ruído administrativo; alinhado com legislação |
| Agrupamento de classes raras | Melhora generalização; reduz overfitting |
| Split temporal | Previne data leakage; monitora dataset drift |
| Threshold operacional: TBD | Será definido após simulação operacional (Recall vs Precision) |

---

## Conclusão
Este projeto demonstra como Ciência de Dados aplicada a Seguros pode apoiar decisões estratégicas, combinando:
- Análise exploratória orientada a negócio
- Modelagem interpretável
- Foco em impacto e valor gerado

** Este projeto tem caráter educacional e demonstrativo, utilizando dados públicos para simular um problema real de mercado.**
