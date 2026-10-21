# BIA-02.5 — Concordância interavaliadores e corpus científico gold versionado

**Estado:** implementação experimental; não implica validação estatística ou científica do acervo real dos INCTs.
**Dependência:** adjudicação BIA-02.4, certificação independente de passagens BIA-01 e validação de direitos.

## Objetivo e unidade de observação

O pesquisador define *antes dos votos* um intervalo literal (`passage_id`, `start`, `end`) e duas identidades de avaliadores (`rater_a`, `rater_b`). A unidade é **rótulo de entidade para o mesmo intervalo predefinido** — não uma anotação livre da página inteira. A classe `NONE` significa ausência de entidade **somente naquele intervalo**. Seleção de intervalos deve incluir positivos e negativos conforme protocolo amostral documentado pelos INCTs. Criar tarefas só para discordâncias introduz viés; por isso a API recebe amostras pré-definidas independentemente do resultado.

1. `POST /v1/agreement/tasks` (curator/admin): escolhe intervalo e dois curadores configurados. Par não pode ter mesma identidade; duplicata exata do par+trecho+intervalo é bloqueada.
2. `GET /v1/agreement/tasks?passage_id=<id>`: tarefa visível apenas ao par e administradores; o voto do outro avaliador **não é mostrado antes dos dois terem respondido**, nem para administradores.
3. `POST /v1/agreement/tasks/{id}/vote` (somente identidade atribuída): registra opinião imutável. Tentativa de segundo voto, de terceiro não autorizado ou de pessoa não atribuída falha.
4. `POST /v1/agreement/tasks/{id}/adjudicate`: se os dois discordarem, uma **terceira identidade** decide o rótulo final e escreve motivo, sem apagar os votos originais. Concordâncias não podem ser reescritas.
5. O programa não transforma automaticamente votos ou decisões em anotações científicas. O pipeline de anotação/certificação anterior continua responsável por aprovar rótulos e offsets.

## Métricas honestas de concordância

`GET /v1/agreement/metrics` (admin) retorna:

- contagem total, completa e incompleta de amostras; taxa observada de acordo por **dupla fixa**;
- matriz de confusão de classes, marginais e acordo esperado pelo produto dos marginais;
- **κ de Cohen = (Po − Pe)/(1 − Pe)** quando matematicamente definido; caso não haja exemplos completos ou `Pe=1`, retorna `null`, nunca `0` fictício;
- divergências pendentes de terceira revisão; aviso explícito quanto à representatividade.

Os cálculos são estratificados por dupla fixa (`rater_a`, `rater_b`), sem somar pares heterogêneos num κ espúrio. Não medem concordância de offsets livremente escolhidos e **não** fornecem intervalo de confiança, generalização por instituto ou aleatoriedade de amostra. Para validação real, os INCTs deverão aprovar desenho amostral, cegamento, estratos e tamanho de amostra.

## Dataset gold restrito e reproduzível

`GET /v1/datasets/gold/export` (somente admin) constrói snapshot interno (`bia.scientific_gold.v1`) a partir de passagens certificadas, com permissão de treinamento verificada e sem disputas em aberto. Se uma tarefa de concordância estiver incompleta, discordante sem adjudicação ou tiver rótulo incompatível com as anotações aprovadas do intervalo, a exportação e o treinamento de entidades falham explicitamente. O treino de relações também ignora essas passagens, até serem reconciliadas.

O snapshot contém `document_sha256`, `publication_group`, ID de passagem, página, texto literal, offsets e rótulos aprovados, `split` determinístico (grupo bibliográfico indivisível), número de documentos/publicações e `sha256` do JSON canônico. `validate_gold_snapshot` verifica integridade contra alterações acidentais; hash **não é assinatura de autoria**, nem garante que a literatura ou os rótulos sejam corretos.

A exportação **não** inclui nomes dos revisores, e o banco mantém esses dados para auditoria interna. Não confundir `training_allowed` com licença para redistribuição: o manifesto fixa `redistribution_allowed=false`. Conteúdo institucional pode integrar o snapshot de treinamento local quando autorizado; por isso o endpoint é restrito a administradores e a resposta NÃO deve ser disponibilizada publicamente nem enviada a serviços de IA externos. A página web mostra apenas contagens e hash.

### Exemplo de uso

```bash
# A URL usa serviço local e requer Bearer token configurado em BIA_API_KEYS.
# Identidades são exemplos: substitua por perfis de curadoria válidos.
curl -X POST http://127.0.0.1:8000/v1/agreement/tasks \
  -H 'Authorization: Bearer ADMIN_TOKEN' -H 'Content-Type: application/json' \
  -d '{"passage_id":"PASSAGE_ID","start":0,"end":8,"rater_a":"alice","rater_b":"bruno"}'

curl -H 'Authorization: Bearer ADMIN_TOKEN' http://127.0.0.1:8000/v1/agreement/metrics
curl -H 'Authorization: Bearer ADMIN_TOKEN' http://127.0.0.1:8000/v1/datasets/gold/export
```

## Garantias e limites operacionais

- Nenhum serviço de IA comercial, checkpoint de terceiro ou modelo pré-treinado; treinamento local do classificador permanece próprio.
- Eventos de criação, votos, decisão e exportação registrados em `audit_events`.
- Amostras, votos e decisões persistem no banco; tabelas criadas automaticamente em desenvolvimento. Implementar **migrações Alembic** antes de produção.
- Credenciais Bearer e papéis são provisórios: em produção adotar identidade institucional robusta (OIDC), grupos por projeto, acesso específico por INCT, logs sem conteúdo sensível, criptografia/retention e controles de concorrência/isolamento de transações.
- Os votos independentes permanecem cegos via API; administradores ainda têm acesso direto ao banco em sua infraestrutura. Segregação real de funções exige governança operacional.
- A exportação é um **snapshot calculado no momento da consulta**, não um registro congelado e assinado em armazenamento imutável; arquivamento versionado, consenso de protocolo e assinaturas são próximos passos.
- Deduplicação apenas em agrupamento de datasets; nenhuma fusão de PDFs, remanejamento de direitos ou limpeza destrutiva é realizada.

## Validação e próximo item

O CI deve executar todos os testes existentes, cenários novos de cegamento, identidade, conflito, κ definido/indefinido, consistência das anotações, direitos, integridade e build Next.js. O corpus sintético de teste prova apenas invariantes de software.

**Próximo item: BIA-02.6 — migrações versionadas, concorrência segura, snapshots assinados/imutáveis e protocolo amostral controlado para avaliação por instituto.**
