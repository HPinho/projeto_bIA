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
