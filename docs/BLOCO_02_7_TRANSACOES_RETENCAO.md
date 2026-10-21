# BIA-02.7 — PostgreSQL concorrente, governança documental e prontidão do corpus

**Estado:** protótipo experimental; não homologado para produção, custódia jurídica ou validação científica dos INCTs. O processamento de IA continua local, sem API comercial e sem pesos externos.

## Baseline e entrega

Parte da baseline verde `8c3394c`. Consolida as verificações antigas e adiciona a revisão **0003**, transições condicionais de certificação, controle de retenção/guarda legal, inventário restrito e job PostgreSQL com dois processos reais. As verificações locais executam SQLite; o job `postgres` no GitHub Actions usa PostgreSQL 16 e é obrigatório para validar o comportamento nesse banco.

## Evolução do banco

- `0001`: tabelas originais.
- `0002`: snapshots gold append-only e avaliações institucionais (triggers contra UPDATE/DELETE).
- **`0003`**: tabela `document_governance`, vinculada ao documento por FK/PK. `0003` é uma migração aditiva; documentos, passagens, versões gold e eventos antigos permanecem.
- `upgrade_schema()` mantém transação e lock de migração (SQLite `BEGIN IMMEDIATE`; PostgreSQL `pg_advisory_xact_lock`). Instalações antigas `0002` podem subir para `0003`. Migrações com histórico desconhecido/fragmentado são recusadas.
- Fazer backup e testar restauração antes da migração institucional. Em produção, rodar `python -m bia_core.schema --database-url ...` manualmente e usar `BIA_AUTO_MIGRATE=0`.

## Concorrência e curadoria

`submit_once`, `certify_once` e `reopen_once` executam `INSERT ON CONFLICT DO NOTHING` ou `UPDATE ... WHERE status = ...`, usando **linhas efetivamente alteradas** para distinguir a operação vencedora da operação obsoleta. A API transforma chamadas perdedoras em HTTP 409. Não se registra segundo evento de certificação para um mesmo estado.

No PostgreSQL, anotações, revisão, submissão, certificação e reabertura obtêm `FOR UPDATE` na linha do trecho para sincronizar alterações com as transições. O teste PostgreSQL executa duas requisições de submissão a partir de **processos do sistema operacional distintos**, exige exatamente um HTTP 200, um 409, um estado coerente e apenas um evento de auditoria. O job ainda prova a imutabilidade do snapshot por trigger PostgreSQL.

**Limites atuais:** ainda é necessário testar profundamente competição entre todos os endpoints de relações, adjudicação e anotação, políticas completas de deadlock/retry, administração multiusuário e autenticação institucional. Este bloco não afirma isolamento de produção abrangente.

## Gestão de documentos e direitos

A nova tabela registra estado de acesso (`active` ou `restricted`), guarda legal (`legal_hold`), prazo `retention_until` UTC com timezone de entrada, justificativa, responsável e data. Endpoints exclusivamente de administrador:

| Rota | Efeito |
| --- | --- |
| `PUT /v1/documents/{id}/governance` | define acesso, prazo e guarda legal com justificativa; no PostgreSQL usa lock da linha do documento |
| `GET /v1/documents/{id}/governance` | status e regras, sem revelar o PDF |
| `GET /v1/corpus/readiness` | contagens agregadas de autorizações, trechos certificados, retenção vencida e guarda legal |
| `GET /v1/corpus/manifest?offset=0&limit=100` | metadados internos paginados, direitos e impedimentos de prontidão; sem PDF/textos |

Exemplo de restrição (admin):

```json
{"access_status":"restricted","legal_hold":true,"retention_until":"2028-01-01T00:00:00+00:00","reason":"Suspensão formal do acesso e guarda das evidências originais"}
```

**Restrições de segurança:**

1. `restricted` bloqueia acesso de perfis não administradores, desativa de imediato a permissão de treinamento e remove o documento de buscas e datasets futuros. A recuperação de snapshots anteriores também fica bloqueada por direitos.
2. Tornar o documento novamente `active` **não devolve** a autorização de treinamento. A autorização precisa de processo jurídico/técnico próprio. A guarda legal não pode ser desativada sem `release_hold=true` explícito, com nova justificativa auditada.
3. O prazo de retenção gera **pendência de revisão**, não remoção automática. Esta versão **não apaga PDFs, banco, backups, checkpoints ou evidências**. Destruição física, política de preservação, incidentes e direitos de titulares necessitam fluxo institucional homologado.
4. A revogação impede novos usos no treino e a exposição futura dos snapshots, mas **não desfaz aprendizado de checkpoints já treinados**, nem interrompe necessariamente um job que já capturou um dataset em memória. Esses artefatos exigem inventário, quarentena e eventual retreinamento supervisionado.
5. O inventário não equivale a licença de redistribuição, aceite dos INCTs, corpus cientificamente representativo nem validação da qualidade das extrações. Não há coleta automatizada de conteúdo protegido por assinaturas.

## Testes e como executar

```bash
python -m pytest -q
# Em ambiente com serviço PostgreSQL 16 e credenciais descartáveis:
BIA_TEST_POSTGRES_URL='postgresql+psycopg://.../bia_test' python -m pytest -q tests/test_postgres_integration.py
```

O CI possui jobs `test`, `frontend` e `postgres`. Um resultado verde exige os três. Se o serviço PostgreSQL não estiver disponível localmente, os testes PostgreSQL são marcados como *skipped*, **não** considerados aprovação de PostgreSQL.

## Próximo item: BIA-02.8

Planejar isolamento de avaliações entre institutos, autenticação federada com papéis reais, auditoria institucional verificável e pré-seleção de um corpus de materiais autorizado por especialistas para uma primeira medição de precisão/recall. Nenhuma dessas etapas é certificada por esta entrega.
