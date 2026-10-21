# Bloco 01 — Rodada 02: dataset adjudicado e treinamento auditável

Baseline confirmado: `531c8d9aaac0c14d61cc4fd26b0906a0b1871065` (GitHub Actions `bIA core CI` verde).

## Problema corrigido

O pipeline anterior gerava `O` (não entidade) em todos os caracteres que não recebiam anotações **aprovadas**, mesmo sem garantia de que o restante da passagem tivesse sido inspecionado. Isso representa um falso negativo de curadoria e contamina a supervisão.

## Entregas

- `passage_reviews`: tabela independente de certificação integral, compatível com bancos bootstrap já criados; estados `draft`, `submitted`, `certified`.
- Fluxo de submissão apenas quando não há pendências, certificação com segunda identidade, imutabilidade enquanto submetido/certificado e reabertura auditada.
- Treinamento usa exclusivamente passagens integralmente certificadas de documentos com permissão explícita de treino, inclusive trechos marcados como realmente sem entidades.
- Segmentação ajusta fronteiras de janelas para preservar rótulos BIO; erros de capacidade de sequência são explícitos.
- Separação por documento, ajuste de tokenizador apenas no conjunto de treino e avaliação de entidades inteiras com precisão/recall/F1 de validação quando aplicável.
- Manifesto com `dataset_sha256`, pesos e tokenizador, e snapshot com anotações aprovadas e atores de certificação.
- Consulta de estado da curadoria, API/UI para revisão de trechos e testes de regressão para os invariantes.

## Não implementado nesta rodada

- Qualificação científica de extração por especialistas reais, OCR de figuras e tabelas complexas, ontologia completa de materiais, relação causal material–processo–propriedade, modelos de linguagem grandes, autenticação federada, migrations Alembic, filas persistentes.
- A segurança atual ainda é apenas adequada a ambiente de desenvolvimento restrito; não publicar acesso a corpus institucional no endereço público.

## Critérios verificáveis

1. O treino falha sem documentos explicitamente autorizados e passagens certificadas.
2. O mesmo pesquisador não pode submeter e certificar o trecho.
3. Anotações pendentes impedem a submissão; trechos submetidos/certificados não podem ser modificados.
4. Reabrir certificação remove o trecho de **novos** treinamentos; pesos já produzidos continuam versionados e precisam de governança posterior.
5. Trechos certificados sem entidades geram exemplos negativos legítimos.
6. Nenhuma entidade é dividida entre janelas de treinamento; janelas inviáveis falham sem perda silenciosa.
7. Treino e validação usam documentos disjuntos, incluindo o vocabulário.

## Próxima entrega recomendada: BIA-02.1

Extração supervisionada de **relações material–processo–condição–propriedade**, sempre ancoradas em spans e páginas verificados, começando pela ontologia e recorte científico aprovados pelos três INCTs. Um atributo relacional sem fonte deve ser `insufficient_evidence`, nunca inventado.
