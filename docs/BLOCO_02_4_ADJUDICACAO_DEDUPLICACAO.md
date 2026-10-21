# BIA-02.4 — Adjudicação de divergências e deduplicação bibliográfica

## Escopo entregue

- **Revisão independente tripartite**: A registra uma alternativa (classe e offsets); B registra sua alternativa no **mesmo trecho original**; C, obrigatoriamente diferente de A e B, adjudica `first`, `second` ou `neither` com justificativa persistida. O registro guarda as duas hipóteses, autores, decisão e referência da anotação criada.
- Trechos só aceitam divergências quando em rascunho. Uma divergência sem resolução **bloqueia submissão e certificação**; assim não se gera `O` negativo falso por falta de consenso. Uma escolha aceita gera anotação aprovada em nome do proponente, revisada pelo adjudicador, sem sobrepor uma anotação existente. `neither` não insere entidade. O caso fica auditável; novas alterações exigem reabertura do trecho.
- **Diferentes versões bibliográficas**: sinais fortes `same_normalized_doi` e sinais para revisão `same_normalized_title_and_year` (título coincidente após normalização conservadora). SHA-256 idêntico de PDF já era bloqueado na ingestão. DOI deve ser válido e normalizado antes de persistir.
- Decisão de par `same_publication` ou `distinct_publication` precisa de justificativa e de curador **diferente dos responsáveis pelos uploads**. Só um parecer por par é admitido na versão inicial: divergências posteriores devem ficar para o processo de recurso (futuro), não sobrescrever a decisão histórica.
- **Nunca funde nem elimina PDFs**, anotações, direitos, citações, páginas ou IDs. Mesmo parecer de publicações equivalentes preserva as duas evidências e direitos individuais.
- Trechos de PDFs de uma publicação confirmada ou com DOI normalizado idêntico permanecem juntos no mesmo grupo dos splits da extração de entidades e da classificação de relações. Títulos parecidos **não** vinculam agrupamentos sem parecer explícito. Um parecer `distinct_publication` **não** separa DOIs idênticos entre splits: preferimos agrupamento conservador contra vazamento.

## Rotas e papéis

Todos os endpoints abaixo exigem papel `curator` ou `admin`. O upload de bibliografia requer direitos cadastrados e os documentos institucionais nunca são enviados a viewers.

| Método | Rota | Finalidade |
|---|---|---|
| POST | `/v1/adjudications` | Abrir divergência com `passage_id` e `alternative: {label,start,end}` |
| POST | `/v1/adjudications/{id}/second` | Segundo curador envia `{label,start,end}` |
| POST | `/v1/adjudications/{id}/resolve` | Terceiro curador envia `resolution` e `reason` (10 a 500 caracteres) |
| GET | `/v1/passages/{id}/adjudications` | Histórico de alternativas e decisões |
| GET | `/v1/documents/{id}/duplicates` | Pares candidatos, sinais e pareceres |
| POST | `/v1/bibliography/review` | Par de IDs, veredito e justificativa |
| GET | `/v1/curation/summary` | Contadores de divergências abertas e pares revisados |

## Segurança científica e limites

1. **Nenhum caso sintético usado nos testes representa avaliação científica real**. Um parecer tripartite por si não substitui a política de amostragem e validação que deve ser definida com os três INCTs.
2. Somente DOI normalizado ou título **exatamente igual depois da normalização**, com mesmo ano, geram sugestão; não são contemplados registros em outros idiomas, variações substantivas de título, erratas ou traduções. O sistema evita fazer falsos merges de alto custo.
3. A equivalência humana agrupa os splits em próximas execuções de treinamento; checkpoints existentes **não são reescritos**. O `dataset_snapshot` continua identificando a versão de cada execução via hash, e novos grupos aparecem nos próximos snapshots.
4. O bootstrap cria tabelas aditivas via SQLAlchemy. **Antes de produção**: migrações Alembic, transações concorrentes/locks de adjudicação, versionamento de recursos e correções de parecer com histórico em vez de sobrescrita. Revisões concorrentes em múltiplos workers não são garantidas por este protótipo.
5. Duas interpretações são preservadas com seus offsets originais; não são usadas como "labels verdadeiros" antes da decisão tripartite. Casos pendentes impedem certificação do trecho.
6. O banco local é uma prova de conceito; tokens Bearer estáticos e acesso institucional amplo para curadores precisam ser substituídos por RBAC real e identidade institucional.

## Verificação e próximo bloco

`python -m pytest -q` e GitHub Actions (`test` e `frontend`). Novos cenários cobrem três identidades, `neither`, offsets, conflitos, direitos, DOI, agrupamento por publicação e prevenção de fusão.

**Próximo bloco BIA-02.5**: medir concordância interavaliadores, exportar dataset adjudicado com esquema versionado e trilhas de conflito auditáveis. Requer decisão dos pesquisadores sobre protocolo de anotação e critérios de referência.
