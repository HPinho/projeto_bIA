# BIA-02.6 — Persistência versionada, concorrência, snapshots e avaliações INCT

**Estado:** protótipo de pesquisa; ainda não homologado para uso institucional em produção ou validado cientificamente pelos INCTs.

## Migrações versionadas

`bia_core/schema.py` possui revisões sequenciais `0001` (esquema histórico) e `0002` (snapshots e avaliações institucionais). O controle fica em `bia_schema_versions`.

- Banco **novo**: cria estrutura inicial, instala a revisão de snapshots e registra as duas revisões.
- Banco **antigo** vindo das versões anteriores: reconhece as tabelas históricas, preserva seus registros e acrescenta apenas as novas tabelas e triggers. Migração interrompe instalação com esquema legado incompleto ou versão desconhecida.
- SQLite: `BEGIN IMMEDIATE` serializa migrações concorrentes e permite rollback do DDL; PostgreSQL: `pg_advisory_xact_lock` transacional evita migrações simultâneas.
- Rodar explicitamente antes de iniciar a API: `python -m bia_core.schema --database-url 'sqlite:///./data/bia.sqlite3'` (ou a URL PostgreSQL apropriada).
- Depois de migrar, configurar `BIA_AUTO_MIGRATE=0` para exigir esquema já atualizado em cada inicialização. Sem essa opção, a API migra automaticamente para conveniência **somente em desenvolvimento**.

**Pré-requisito institucional:** backup verificado do banco e do armazenamento de PDFs, período de manutenção, testes de restauração e homologação PostgreSQL com múltiplas instâncias. Não há downgrade automático. O protocolo atual foi testado localmente com SQLite; PostgreSQL requer teste de integração específico com serviço real.

## Transações concorrentes

A duplicação de amostras, votos, adjudicações, pares bibliográficos e avaliações por instituto usa restrições únicas do banco e conversão de conflitos `IntegrityError` para HTTP 409. Os votos são gravados uma única vez por identidade atribuída. Também há teste de duas requisições paralelas de voto para a mesma amostra, demonstrando uma escrita bem-sucedida e uma resposta 409, sem duplicação.

**Limitação:** a revisão otimista de todas as operações de curadoria, especialmente o ciclo submit/certify/reopen, ainda não foi certificada sob alta concorrência PostgreSQL. Não tratar esse bloco como garantia de isolamento de produção; será necessário endurecer transições de estado com atualizações condicionais e testes em múltiplos processos.

## Snapshots gold imutáveis

As rotas `POST /v1/datasets/gold/snapshots`, `GET /v1/datasets/gold/snapshots` e `GET /v1/datasets/gold/snapshots/{sha256}` exigem perfil administrador. A criação só aceita material efetivamente permitido para treino, certificado e com conciliação de anotações concluída. O conteúdo é serializado em ordem canônica e identificado por SHA-256. Repetir o mesmo conteúdo reutiliza o hash sem criar nova versão.

A versão `0002` instala triggers SQL que recusam `UPDATE` e `DELETE` da tabela `dataset_snapshots`, inclusive fora da API. O sistema verifica integridade do conteúdo a cada acesso. Se algum documento perder a permissão de treinamento ou for excluído, a recuperação do texto de um snapshot anterior é **bloqueada**, mesmo que a hash ainda exista; a lista de metadados continua auditável. Leitura e criação emitem eventos no log.

**Limitações:** hashes e triggers não são assinaturas digitais nem protegidos de administradores com capacidade de desabilitar triggers; também não resolvem prazos legais de retenção ou direito ao apagamento. Não exportar arquivos com restrições de redistribuição. Governança de retenção, criptografia em repouso e backups devem ser tratados antes de produção.

## Protocolo de avaliação dos três INCTs

Identificadores estáveis, sem inferência pelo assunto do artigo:

- `inovaligno` — INCT InovaLigno
- `polissacarideos` — INCT Polissacarídeos
- `circularidade` — INCT Circularidade

A configuração da credencial precisa declarar explicitamente a lista `institutes`, por exemplo:

```json
{
  "local-alice-token": {"actor": "alice", "role": "curator", "institutes": ["inovaligno"]},
  "local-admin-token": {"actor": "admin", "role": "admin", "institutes": []}
}
```

O avaliador não pode ser nenhum dos dois anotadores gold nem a terceira pessoa que resolveu a divergência. O gold precisa estar completo e coerente com o trecho **atualmente certificado**. As avaliações são registradas por instituto, tarefa e hash de certificação; não geram exemplos de treinamento e não revelam o gabarito na resposta de gravação.

- `GET /v1/institutes/{institute}/protocol` — exige afiliação e curadoria.
- `POST /v1/institutes/{institute}/assessments` com `{"task_id":"...","label":"NONE"}` — voto institucional separado, irreversível para a mesma certificação.
- `GET /v1/institutes/assessments/report` — exclusivo do administrador, calcula acordo exato com o gold, matriz de confusão, cobertura e recall descritivo por rótulo. Informa `null` quando não há exemplos; inclui hash canônico do relatório.

Reabrir o trecho remove a avaliação antiga do relatório corrente, mesmo após recertificação com nova hash. O banco conserva o registro histórico e permite outra avaliação sob a nova certificação.

**Metodologia científica pendente:** amostragem pré-especificada por área/material, balanceamento por instituto, avaliação de erros, desenho amostral e aprovação dos especialistas. A concordância com um gold sintético não mede a precisão do classificador neural e não representa validação científica institucional.

## Testes e continuidade

`python -m pytest -q` inclui testes de migração legada, rollback de migração parcial, ausência de triggers, imutabilidade SQL, direitos revogados, duplicidade, controle de acesso por instituto, isolamento de avaliadores, relatório sem vazamento e concorrência de votos. O workflow CI deve testar Python e Next.js antes de considerar o commit verde.

**Próximo bloco proposto: BIA-02.7 — infraestrutura de produção e coleta científica**, incluindo transições de estados condicionais, testes PostgreSQL reais, controle de retenção e aquisição de amostras científicas autorizadas para validação tripartite.
