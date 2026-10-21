# BIA-02.1 — relações científicas rastreáveis (rodada 03)

## Escopo entregue

Núcleo local de relação orientada (`ScientificRelation`) entre **entidades anotadas por humanos**. Os fatos publicados no grafo devem passar por certificação integral do trecho de origem, aprovação dos dois argumentos, proposta explicitamente selecionada por um curador, revisão independente da própria relação e confirmação de que a certificação do trecho continua vigente.

Esta etapa **não implementa um classificador neural de relações**. O modelo neural anterior continua restrito à extração experimental de entidades. O gerador de sugestões utiliza apenas pares de entidades aprovadas e próximas (limite conservador de 300 caracteres); proximidade textual não prova relação científica, causalidade nem desempenho de material.

## Assinaturas permitidas e direção

| Predicado | Origem | Destino |
| --- | --- | --- |
| `MATERIAL_HAS_PROPERTY` | MATERIAL | PROPERTY |
| `MATERIAL_PROCESSED_BY` | MATERIAL | PROCESS |
| `PROCESS_HAS_CONDITION` | PROCESS | CONDITION |
| `PROCESS_HAS_QUANTITY` | PROCESS | QUANTITY |
| `PROPERTY_HAS_QUANTITY` | PROPERTY | QUANTITY |
| `QUANTITY_HAS_UNIT` | QUANTITY | UNIT |
| `MATERIAL_HAS_APPLICATION` | MATERIAL | APPLICATION |
| `MATERIAL_CONTAINS` | MATERIAL | MATERIAL |

Estes nomes são um **vocabulário técnico provisório**. A interpretação química e sua ontologia precisam de discussão com representantes dos três INCTs. Não se infere automaticamente causalidade, aumento/diminuição, unidade física compatível ou generalização para outros materiais.

## Protocolo verificável

1. Cadastrar o PDF respeitando a base de direitos. Documentos institucionais permanecem invisíveis a visitantes comuns.
2. Anotar entidades usando offsets literais; outra identidade revisa as entidades; submeter e certificar integralmente o trecho com uma segunda identidade.
3. O curador chama `GET /v1/passages/{id}/relation-candidates` para receber *hipóteses não verificadas* (pares de entidades próximas).
4. Curador escolhe uma combinação cientificamente relevante e propõe `POST /v1/relations` com `passage_id`, IDs de duas anotações aprovadas, `relation_type`, `evidence_start`, `evidence_end`. A evidência precisa conter ambos os argumentos no texto original, respeitando comprimento máximo de 1000 caracteres.
5. Outro curador, com identidade diferente da do proponente, aprova ou rejeita via `POST /v1/relations/{id}/review` e `{"decision":"approved"}` ou `{"decision":"rejected"}`. A decisão fica auditada, independentemente das outras revisões de entidade ou trecho.
6. `GET /v1/passages/{id}/relations` mostra ao curador relações pendentes, rejeitadas, aprovadas ou obsoletas.
7. `GET /v1/relations/search?q=...` mostra a qualquer papel autorizado **somente relações aprovadas e ainda vinculadas à certificação atual**, filtrando documentos por direitos. Cada resultado inclui título, DOI quando disponível, página, trecho original, offsets dos argumentos e identidades do proponente e revisor. Suporta também `relation_type`, `limit`. Sem correspondência retorna `insufficient_evidence`.
8. Ao reabrir um trecho para corrigir suas anotações, as relações de sua certificação anterior deixam de ser publicáveis imediatamente. Mesmo após recertificar, uma nova relação e revisão são exigidas. Registros antigos ficam preservados para auditoria.

## Limites e qualidade

- O campo `scientifically_validated=false` da busca assinala que revisão local ainda não equivale à validação científica externa. Os dados de teste são **sintéticos** e não representam alegações sobre materiais reais.
- O sistema **não** interpreta PDFs com OCR, figuras complexas ou tabelas integralmente. A granularidade de origem atual é a página/trecho gerado pela extração de texto.
- Não se devem usar relações rejeitadas como negativos de treinamento de maneira automática: rejeição pode ter múltiplos motivos científicos. Uma futura taxonomia de motivos de rejeição permitirá criar negativos explícitos sem fabricar dados.
- Ainda não existe normalização de unidades, ontologia química validada, deduplicação semântica entre artigos ou detecção sistemática de contradições.
- SQLite e `create_all` são um bootstrap local; antes de produção é necessário Alembic, banco/filas adequados, controle institucional de identidades, auditoria mais forte, backup e autorização científica/documental formal.

## Critérios para aceitar BIA-02.1

- Não aceitar proposta sem certificação completa e entidades aprovadas, nem argumentos em passagens distintas.
- Rejeitar assinatura de tipos incompatíveis e offsets que não sustentem ambos os argumentos.
- Bloquear autorrevisão, publicação pendente/rejeitada e exposição de dados institucionais.
- Invalidar relações antigas após reabertura/recertificação sem apagar histórico.
- Devolver sempre uma evidência literal rastreável na consulta, sem inventar DOIs ou resultados experimentais.
- Preservar regressões do Bloco 01 (11 testes originais) e passar os testes novos.

## Próximo item — BIA-02.2

**Dataset de relações e classificador neural próprio**, condicionado à definição pela equipe científica dos motivos de rejeição, direcionalidade, granularidade de evidência, critérios de criação de negativos verdadeiros e qualidade do corpus. Construir treino/validação/teste agrupados por documento, pesos iniciados do zero, métricas de relação por par e predicado e sem publicação automática de relações preditas.
