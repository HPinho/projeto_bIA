# bIA Bloco 01 — Arquitetura e limites

## Missão

Dar suporte ao levantamento e à análise de literatura científica em materiais, inicialmente voltada às interfaces entre INCT InovaLigno, INCT Polissacarídeos e INCT Circularidade. O recorte e as perguntas científicas **ainda dependem de decisão dos pesquisadores**.

## Pipeline disponível

1. Cadastramento de PDF digital com título, DOI opcional, ano opcional, declaração de base de direitos e confirmação **separada** da autorização para treinamento.
2. SHA-256 e armazenamento local privado; extração de texto por página com `pypdf`. Documentos sem texto entram em `needs_review`; não são classificados como vazios.
3. Pesquisa BM25 escrita no projeto. Saídas documentam IDs, título, DOI, página, trecho e score. Evidência ausente produz `insufficient_evidence`.
4. Curadores anotam offsets Unicode do texto extraído. Outra identidade precisa aprovar ou rejeitar a anotação. Depois, o trecho integral deve ser submetido e certificado por identidade distinta; inclusive a ausência de entidades deve ser confirmada. Reabertura invalida a elegibilidade de treinamento. Todos os eventos são auditados.
5. Treinamento local CPU com tokenizador Unicode treinado no corpus permitido, encoder Transformer (atenção multi-head, MLP, embeddings e normalização) implementado no repositório, inicializado de pesos aleatórios. PyTorch fornece apenas operações, gradientes e otimização.
6. Checkpoint de pesos, vocabulário, manifesto de métricas e snapshot incluindo todas as anotações aprovadas e revisores, com hashes e split agrupado por SHA-256 do documento. O vocabulário só observa treino, jamais o conjunto de validação ou teste; treinamento admite apenas trechos integralmente certificados e autorizados.
7. Inferência experimental retorna offsets e segmentos **exatamente recuperados do texto original**. Nunca atribuir validação científica com base num smoke test.

## Modelo e vieses

O tokenizador **inicial** é em caracteres, não BPE. Isso permite testar UTF-8/Unicode e offsets sem componentes externos, mas é menos eficiente. O encoder **não** é um LLM conversacional. O desempenho dependerá de corpora anotados e revisão tripartite. Os resultados são provisórios.

## Segurança e limitações antes de produção

- Token Bearer com identidade e papel configurados por variável de ambiente; sem token configurado o acesso científico fica bloqueado (fail-closed).
- Papéis: viewer e researcher (consultas a documentos não institucionais), curator (importação, anotação, revisão e documentos institucionais), admin (também inicia treinamento). Acesso institucional amplo a curadores é uma **política provisória**: implementar grupos/organizações/permissões por projeto antes de uso real.
- Upload limita o tamanho e valida PDF, mas ainda não há antivírus/sandbox de parsing. PDFs devem ser confiáveis na fase local.
- CORS configurado apenas para frontend local por padrão; nunca compartilhar tokens no Git.
- Armazenamento privado fora do Git; o serviço não expõe endpoint de download do PDF original.
- Jobs usam executor local em memória. Reinícios interrompem a execução. Em produção, adotar fila persistente e quotas por equipe.
- Criação automática de tabelas via SQLAlchemy é somente para bootstrap; antes de produção implantar Alembic com migrações versionadas.
- Pesquisa reavalia corpus em memória a cada consulta; substituir por índice persistente para escala.
- PDF digital simples apenas, sem OCR científico, extração de gráficos ou tabelas de alta confiabilidade. Revisão especializada é indispensável.
- Métricas atuais são de correspondência exata de entidade BIO em validação interna, **não** de concordância entre extratores e especialistas. Sem exemplos de validação, são `null`.
- As janelas de caracteres deslocam suas fronteiras para não cortar entidades BIO; entidades acima de `max_length` são rejeitadas explicitamente, sem descarte silencioso. Um corpus científico grande exigirá batching e monitoramento de capacidade.
- Submissão e certificação são atos de curadoria dentro do protótipo, não uma alegação de validação científica externa. Antes de produção, registrar protocolos de avaliação e acordos entre os INCTs.
- A tabela `passage_reviews` é nova e pode ser criada pelo bootstrap SQLAlchemy, mas **não** há migrações Alembic de produção.
- O dataset de avaliação deve ser definido por pesquisadores, com amostras não sintéticas e aprovação de política de direitos de uso.

## Próximo bloco

Bloco 02: extração supervisionada de relações material–processo–condição–propriedade e avaliação científica por instituto. Pré-requisito: recorte temático, treinamento permitido, protocolo de anotação, baseline verde e critérios mínimos de aceitação acordados.

### Bloco 02.1: grafo curado de relações (rodada 03)

Relações direcionais usam duas entidades aprovadas no mesmo trecho, offsets literais de evidência e um identificador criptográfico da **versão de certificação**. Sugestões heurísticas são apresentadas como hipóteses, não como resultados do modelo. A consulta é restrita a relações aprovadas por revisor independente, certificação vigente e direitos de acesso compatíveis. A reabertura do trecho invalida o grafo derivado sem apagar o registro auditável. A próxima frente BIA-02.2 trata de treinamento de relações com dados explicitamente validados, não de treinar em negativas presumidas. Consulte `docs/BLOCO_02_1_RELACOES.md`.

### BIA-02.2 — Módulo de aprendizado de relações

`bia_core/relation_learning.py` adiciona classificador char-Transformer próprio, com pesos locais, entradas de offsets de sujeito/objeto e tipo do predicado. `RelationNegativeAssessment` guarda negativos **expressamente** confirmados como ausentes (rejeição genérica não vale como negativo). `RelationModelRun` isola jobs e artefatos do extrator NER. Predições são apenas hipóteses; não são importadas ao grafo verificado. Veja [protocolo BIA-02.2](BLOCO_02_2_APRENDIZADO_RELACOES.md). Próxima implementação: **BIA-02.3 — avaliação e versionamento de datasets científicos**.

### BIA-02.3 — Avaliação por publicação e integridade de resultados

O classificador de relações agora agrupa documentos com DOI normalizado idêntico e passagens idênticas nos mesmos splits. `bia_core/evaluation.py` calcula matriz de confusão, F1 por relação e documento, erros sem texto sensível e métricas `null` para suportes insuficientes. Artefato `evaluation_report.json` é vinculado ao dataset congelado via SHA-256 e consultável apenas por administradores após verificação dos hashes. Não existe validação científica em corpora reais nesta fase. Consultar [protocolo BIA-02.3](BLOCO_02_3_AVALIACAO_CIENTIFICA.md). Próxima frente: **BIA-02.4, adjudicação tripartite e duplicatas bibliográficas**.

### BIA-02.4 — Adjudicação e vínculos bibliográficos

A adjudicação preserva as alternativas de duas identidades, exige uma terceira identidade para decidir e bloqueia certificação diante de divergências em aberto. Pares de publicações com DOI idêntico ou título+ano normalizados produzem apenas **sugestões de duplicatas**; um revisor humano diferente dos responsáveis pela ingestão fundamenta sua decisão. A confirmação de versões equivalentes liga seus splits de aprendizado, mas jamais une nem descarta PDFs, seus direitos ou suas evidências. Consulte [o protocolo BIA-02.4](BLOCO_02_4_ADJUDICACAO_DEDUPLICACAO.md). Próxima frente: **BIA-02.5 — concordância interavaliadores e corpus adjudicado**.

### BIA-02.5 — Dupla avaliação cega e gold auditável

`agreement_tasks`, `agreement_votes` e `agreement_resolutions` documentam intervalo definido antes dos votos, duas identidades fixas, respostas ocultas até finalização e decisão de terceira identidade para discordância. `bia_core/agreement.py` calcula κ de Cohen **por dupla fixa**, não agrega indivíduos diferentes e usa `null` quando indefinido. A certificação científica não é presumida a partir de acordo estatístico. `source_dataset` recusa amostras com tarefas pendentes ou incompatíveis com a anotação aprovada; treino de relações também ignora essas passagens. Um snapshot interno restringe texto a documentos autorizados para treinamento, agrupa splits por publicação e guarda `sha256` canônico. Exportação exclusiva de admin, sem licença de redistribuição. Consultar [BIA-02.5](BLOCO_02_5_CONCORDANCIA_GOLD.md). Próximo: **BIA-02.6 — migrações, transações concorrentes e versões imutáveis**.

## BIA-02.6 — Persistência versionada e avaliação tripartite

- Banco legado passa por revisões controladas `0001 -> 0002` (`bia_core/schema.py`), com rollback transacional e bloqueio de startup quando a migração explícita é exigida.
- `DatasetSnapshot` persiste o corpus gold autorizado em JSON canônico com SHA-256; triggers SQL recusam UPDATE/DELETE, e revogação atual de direitos bloqueia leitura histórica via API.
- `InstituteAssessment` registra observação independente por instituto, tarefa e versão da certificação. Não influencia o gold nem o treinamento; relatórios por instituto usam somente trechos com certificação vigente.
- Restrições UNIQUE e `IntegrityError` tratado para operações críticas de concordância e bibliografia; alta concorrência de todos os estados e PostgreSQL distribuído continuam pendentes para produção.
- Detalhes, interfaces, riscos e próximos testes: [BIA-02.6](BLOCO_02_6_INFRAESTRUTURA.md).


## Extensão BIA-02.7

A migração `0003` instala `document_governance`: restrição de acesso e treinamento, prazo de revisão da retenção e guarda legal sem exclusões automáticas. Transições de submissão/certificação/reabertura usam operações SQL condicionais; PostgreSQL usa lock de linha do trecho nas mutações de curadoria. O CI possui testes transacionais com serviço PostgreSQL 16 e processos distintos. Inventário de prontidão é **somente administrativo**, sem conteúdo de PDFs. Ver [BLOCO_02_7_TRANSACOES_RETENCAO.md](BLOCO_02_7_TRANSACOES_RETENCAO.md).
