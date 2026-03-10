# Metricas de Avaliação

## COMO E QUANDO USAR CADA MÉTRICA DE AVALIAÇÃO - CASOS PRÁTICOS EM BANCO

| Métrica | Quando Usar | Por Que Usar | Caso Prático em Banco |
|---------|-------------|--------------|----------------------|
| **Acurácia** | Classes balanceadas | Medida geral de performance | Aprovação de cartões (taxa similar de aprovados/negados) |
| **Precisão** | Custo alto de falsos positivos | Evitar aprovar clientes inadimplentes | Prevenção de fraudes (evitar bloquear transações legítimas) |
| **Recall** | Custo alto de falsos negativos | Evitar deixar fraudes passarem | Detecção de lavagem de dinheiro (capturar todas as transações suspeitas) |
| **F1-Score** | Balancear precisão e recall | Quando ambas métricas são importantes | Scoring de crédito (equilíbrio entre risco e oportunidade) |
| **AUC-ROC** | Comparar modelos globalmente | Avaliar performance em todos os thresholds | Seleção do melhor modelo de crédito |
| **Matriz de Confusão** | Análise detalhada dos erros | Entender tipos específicos de erro | Diagnóstico de performance do modelo |

### FORMAS MAIS FÁCEIS DE INTERPRETAR AS MÉTRICAS DE AVALIAÇÃO

Imagine que vc tem um dataset com 100 dados de transação, 99 são normais e uma é fraude. Se vc escrever um código só para printar 'não é fraude' você vai ter uma acurácia de 99%, porque ela mostra o quanto vc tá prevendo correto em relação a todas tentativas, mas sabemos que isso não faz sentido, então o dado sofrerá com o desbalanceamento

Precisão é a taxa de positivos corretos, queremos diminuir ao máximo os falsos positivos, exemplo: classificar se o vídeo é apropriado para criança

Recall, taxa de negativos corretos, exemplo, falar que uma pessoa não tem cancer, quando na verdade ela tem

f1 score: equilibra precisão e recall