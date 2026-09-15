# Gen AI

---
# Pontos cobertos pelo questionario abaixo:
- Embeddings e banco vetorial ✅


---
## Em um sistema de RAG, qual é a função dos embeddings e do banco vetorial? E por que não basta simplesmente enviar todos os documentos para o LLM de uma vez?
Embeddings são vetores numéricos que representam semanticamente textos ou outros conteúdos. No RAG, os documentos são primeiro divididos em chunks, depois esses chunks são convertidos em embeddings e armazenados em um banco vetorial. Quando chega uma pergunta, ela também é transformada em embedding e o sistema busca os chunks semanticamente mais próximos. Esses trechos relevantes são enviados ao LLM como contexto. Não se envia todo o acervo porque isso aumenta custo, latência, pode ultrapassar a janela de contexto e introduzir muita informação irrelevante.