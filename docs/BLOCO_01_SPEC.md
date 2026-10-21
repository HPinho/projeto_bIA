# bIA — Bloco 01: Núcleo científico independente

**Versão do planejamento:** 0.1  
**Status:** especificação técnica; implementação ainda não iniciada neste documento  
**Diretriz:** modelos e pesos próprios, inicializados aleatoriamente e treinados com dados admissíveis; bibliotecas externas restritas à infraestrutura computacional, interface, armazenamento e leitura de arquivos; nenhuma dependência de APIs de IA comerciais nem de modelos pré-treinados.

## 1. Fontes e decisões

Este bloco concretiza o fluxo dos dois documentos-base fornecidos para a bIA:

- **Proposta bIA (2 páginas):** escolher recorte temático interdisciplinar entre INCT InovaLigno, INCT Polissacarídeos e INCT Circularidade; coletar metadados por buscas científicas; construir acervo de PDFs com direitos de uso respeitados; extrair e validar informações com pesquisadores; oferecer interface de consulta científica.
- **Fluxograma bIA (1 página):** organizar equipe e infraestrutura, preparar acervo e metadados, desenvolver IA internamente, obter aprovação de especialistas e testar qualidade científica, segurança e desempenho.

Decisões técnicas **propostas**, não fatos estabelecidos nos documentos: arquitetura Python/PyTorch/PostgreSQL, tokenizador próprio, índice BM25, modelo neural pequeno desde o zero, APIs internas e critérios de testes abaixo.

## 2. Meta do Bloco 01

Entregar um **caminho vertical de ponta a ponta**:

`PDF autorizado → registro e metadados → texto/página/tabela → anotação/validação → índice e busca → treino do primeiro modelo bIA → extração de entidades → consulta com evidências`.

Ao término, um pesquisador deve conseguir cadastrar um documento permitido, inspecionar texto recuperado, corrigir entidades científicas, treinar uma versão reproduzível do modelo e consultar resultados com identificação precisa da fonte. A saída inicial será prioritariamente **estruturada e fundamentada**, não uma conversa livre de LLM.

### Dentro do escopo

1. Backend, banco e armazenamento documental.
2. Ingestão de PDFs digitais e metadados bibliográficos, com fluxo de revisão para PDFs difíceis.
3. Normalização e segmentação rastreáveis por página e intervalo de caracteres.
4. Busca lexical científica própria com BM25 e filtros por material, tema, ano e fonte, quando esses campos estiverem disponíveis.
5. Vocabulário/ontologia inicial de entidades e anotação supervisionada por especialistas.
6. Tokenizador implementado pelo projeto e treinado somente nos dados admissíveis da bIA.
7. Primeiro modelo de extração científica implementado no projeto, com pesos inicializados aleatoriamente e treinamento local.
8. Avaliação reproduzível e interface web mínima de pesquisa/revisão.
9. Segurança, licenças, rastreabilidade, testes e documentação.

### Explicitamente fora do Bloco 01

- LLM conversacional grande, geração livre de respostas ou planejamento autônomo de pesquisas.
- Afirmação de descoberta de novos materiais ou previsão de propriedades sem validação.
- OCR científico robusto para todos os tipos de PDF e reconhecimento completo de gráficos/equações.
- Treinamento em PDFs sem autorização de uso; redistribuição de documentos protegidos.
- Dependência de serviços de IA externos, download obrigatório de checkpoints de terceiros ou uso de embeddings pré-treinados.

## 3. Pilha técnica e fronteira de independência

| Camada | Proposta | Papel |
|---|---|---|
| Linguagem | Python 3.12+ | Orquestração e desenvolvimento dos componentes próprios |
| Computação | PyTorch (tensor, autograd e kernels CPU/GPU), NumPy | Infraestrutura numérica; **não** utilizar transformers/encoders/pesos pré-treinados |
| API interna | FastAPI + Pydantic | Infraestrutura HTTP e contratos tipados |
| Persistência | PostgreSQL + migrações Alembic | Dados científicos, anotações, auditoria e índices |
| Documentos | Armazenamento local privado com hashes SHA-256 | Originais e derivados versionados; caminho para S3 compatível opcional |
| Parsing | pypdf e pdfplumber (licenças a conferir na versão fixada) | Extração de PDFs textuais e tabelas, nunca inferência IA |
| Interface | Next.js/React | Administração, anotação e consulta |
| Testes/CI | pytest, ferramentas de lint/typecheck, GitHub Actions | Regressões, segurança e reprodutibilidade |

**Regra de independência:** sem SDKs de IA comerciais, chamadas a provedores de inferência, bibliotecas que baixem automaticamente pesos, `from_pretrained`, chamadas a embeddings externos ou modelos de linguagem hospedados. O uso de uma API bibliográfica para **metadados** é assunto separado das APIs de **IA**; no primeiro bloco, a ingestão por exportações CSV/BibTeX/RIS e upload autorizado dispensa essa dependência.

**Atenção:** “modelo treinado do zero” não significa recriar drivers GPU, CUDA, compiladores, redes e sistema operacional. PyTorch será utilizado exclusivamente como base de computação e diferenciação automática. A definição de modelo, tokenização, objetivo de treinamento, pesos, avaliação e inferência serão parte do código bIA.

## 4. Arquitetura proposta

```text
[Upload permitido: PDF + CSV/RIS/BibTeX]
                |
       [Validação de entrada]
                |
   [Registro bibliográfico + SHA256]---->[Armazenamento privado]
                |
        [Extração por página]
                |
   [Normalização / trechos / tabelas]
           |                   |
     [Índice BM25]      [Tela de anotação]
           |                   |
           |            [Conjunto rotulado]
           |                   |
           |        [Tokenizador bIA treinado]
           |                   |
           |          [Modelo bIA do zero]
           |                   |
           +-------[Recuperação + extração]-----+
                                               |
                                   [Resposta com evidências]
                                               |
                                     [Pesquisador revisa]
```

### Organização sugerida de repositório

```text
bia/
├── apps/
│   ├── api/                  # Endpoints HTTP internos e autenticação
│   └── web/                  # Consulta, upload, revisão e anotação
├── packages/bia_core/
│   ├── domain/               # Entidades, tipos, critérios e validações
│   ├── ingestion/            # Importações e PDF textual
│   ├── corpus/               # Segmentação, hashes, spans e proveniência
│   ├── retrieval/            # Tokenização de busca, índice e BM25
│   ├── annotation/           # Esquema BIO e revisão humana
│   ├── tokenizer/            # Treino e codec próprios, sem checkpoints
│   ├── ml/
│   │   ├── layers.py         # Atenção, embeddings e camadas próprias
│   │   ├── encoder.py        # Arquitetura do primeiro encoder bIA
│   │   ├── objectives.py     # Perdas e tarefas supervisionadas
│   │   ├── train.py          # Inicialização e treinamento reproduzível
│   │   ├── infer.py          # Inferência totalmente local
│   │   └── checkpoints.py    # Versões, config, hash e proveniência
│   ├── evaluation/           # Métricas e conjuntos de teste imutáveis
│   └── evidence/             # Trechos originais, DOI, página, validação
├── infra/                    # Compose, migrações, ambientes e deploy
├── tests/                    # Unitários, integração, contrato, ML
├── datasets/                 # Apenas dados permitidos; ignorados no Git
├── docs/                     # ADRs, dicionário, protocolo e resultados
└── .github/workflows/        # CI sem uso obrigatório de GPU
```

## 5. Modelo de dados mínimo

- `document`: id, título, DOI opcional, autores, ano, fonte, tema, status e direitos de uso.
- `document_asset`: document_id, caminho protegido, sha256, tipo, tamanho, data da ingestão.
- `passage`: document_id, página, ordem, texto bruto, texto normalizado, offsets, versão do extrator.
- `table_fragment`: documento, página, célula/linha/coluna quando recuperáveis, texto e status de confiança.
- `entity_span`: passage_id, offsets, categoria, texto original, origem (humano/modelo), versão.
- `relation`: entidade de origem, relação tipada, entidade de destino, referência do trecho e status.
- `annotation_review`: anotador, revisão, decisões, timestamps e justificativa.
- `dataset_snapshot`: hashes dos registros, política de divisão por documento, esquema de rótulos.
- `training_run`: seed, commit, configuração, snapshot, métricas, ambiente e artefatos.
- `model_version`: tokenizador, arquitetura, pesos, validação, hash e estado de implantação.
- `audit_event`: ação, ator, recurso, data e resultado.

**Regras essenciais:** (i) DOI pode faltar; (ii) deduplicação por DOI normalizado **e/ou** hash do arquivo, sem fundir incorretamente versões; (iii) todos os trechos têm documento e página; (iv) não perder texto/valores originais; (v) os dados de teste não podem contaminar treino; (vi) não atribuir relação experimental sem evidência e revisão/medida de confiança.

## 6. Esquema científico inicial para anotação

Entidades provisórias, a confirmar com grupo tripartite:

| Rótulo | Significado | Observação |
|---|---|---|
| `MATERIAL` | Material, biomassa ou componente | Preservar forma original e sinônimos |
| `PROCESS` | Processo, tratamento ou síntese | Não inferir processo ausente |
| `PROPERTY` | Propriedade física, química ou mecânica | Ex.: classe de propriedade, sem inventar valor |
| `QUANTITY` | Valor numérico reportado | Manter separador decimal e sinal originais |
| `UNIT` | Unidade associada à medida | Não converter automaticamente sem regra validada |
| `CONDITION` | Temperatura, tempo, pressão e outras condições | Distinguir método de resultado |
| `APPLICATION` | Uso ou aplicação do material | Capturar apenas o que foi reportado |

Relações propostas para uma etapa supervisionada: `material_has_property`, `property_has_value`, `process_applied_to_material`, `measured_under_condition`, com origem documental obrigatória. Nenhuma categoria ou relação será considerada cientificamente aprovada sem revisão por especialistas.

## 7. Primeiro motor inteligente: plano técnico

### 7.1 Busca local determinística

- Construir tokenização lexical científica que preserve fórmulas, símbolos, unidades e hifens relevantes.
- Criar índice invertido com estatísticas de frequência e comprimento dos documentos.
- Implementar cálculo e testes de ordenação BM25 (parâmetros versionados).
- Adicionar filtro por ano, DOI, instituto, licença, texto completo disponível e revisão.
- Devolver trecho, documento, página, score e motivo/filtros usados.
- Incluir testes com corpus sintético onde a ordem dos resultados é previamente conhecida.

### 7.2 Tokenizador neural próprio

- Implementar um tokenizador subword com algoritmo próprio de treinamento (por exemplo BPE de bytes), versionado e sem vocabulário externo pré-treinado.
- Documentar codificação reversível quando aplicável, mapeamento token→offset e tratamento de caracteres científicos, expoentes, µ, °, subscritos e fórmulas.
- Fixar `tokenizer_version`, corpus e seed; não criar splits de treino/validação/teste por fragmentos de um mesmo artigo.

### 7.3 Primeiro modelo neural bIA

**Proposta inicial:** encoder Transformer pequeno construído no próprio projeto com operações PyTorch básicas, embeddings, atenção multi-head, normalização, MLP e camada de classificação por token. Dimensões sugeridas apenas como ponto de partida configurável: 4 camadas, `d_model=256`, 4 cabeças, sequências até 384 tokens. Essas especificações não implicam desempenho esperado.

1. Inicializar todos os parâmetros por seed, sem pesos de terceiros.
2. Testar sobreajuste intencional em pequeno conjunto sintético para verificar que há aprendizado e que gradientes fluem.
3. Preparar pré-treino opcional por máscara de tokens **somente** quando houver corpus e permissão de uso suficientes.
4. Treinar reconhecimento de entidades BIO com dados anotados; manter avaliação por artigo/documento, não por trechos aleatórios misturados.
5. Salvar pesos, tokenizador, rótulos, splits, commit, configurações e métricas.
6. Expor inferência local com rastreamento de trechos, scores calibráveis e revisão obrigatória para usos sensíveis.
7. Se os dados anotados forem insuficientes, manter o motor lexical e as regras como baseline utilizável; um treino técnico bem-sucedido não é evidência de precisão científica.

### 7.4 Contrato de evidência

Toda entidade extraída ou resposta factual mostrada ao usuário deve manter:

`document_id`, `doi_or_repository_id`, `page`, `passage_id`, `exact_quote_or_span`, `extraction_version`, `review_status`.

Se não houver evidência suficiente, retornar estado `insufficient_evidence`, nunca inventar artigo, DOI, página, experimento ou valor.

## 8. Interfaces e endpoints internos previstos

| Método | Caminho | Resultado |
|---|---|---|
| `POST` | `/v1/documents` | Registrar PDF e metadados permitidos |
| `GET` | `/v1/documents/{id}` | Consultar documento, situação e direitos |
| `GET` | `/v1/documents/{id}/passages` | Trechos e páginas para revisão |
| `POST` | `/v1/annotations` | Criar rótulos e relações com offsets |
| `POST` | `/v1/annotations/{id}/review` | Aprovar/rejeitar anotação |
| `GET` | `/v1/search` | Busca com score e evidências |
| `POST` | `/v1/models/train` | Iniciar job local autorizado, sem bloquear HTTP |
| `GET` | `/v1/models/runs/{id}` | Estado, artefatos e métricas |
| `POST` | `/v1/extract` | Extração por modelo local versionado |
| `GET` | `/health` | Saúde da aplicação e dependências |

A implementação deverá incluir autenticação, papéis básicos (`admin`, `curator`, `researcher`, `viewer`) e isolamento de documentos conforme licenças e permissões. Jobs de treinamento são executados em worker local, não na thread HTTP. Os nomes de rotas são proposta técnica.

## 9. Sequência de entrega em um bloco integrado

| ID | Frente | Critério de entrega |
|---|---|---|
| BIA-01 | Monorepo e infraestrutura | Ambiente sobe localmente, migrações e CI rodando |
| BIA-02 | Modelo de domínio e permissões | Esquema versionado, acesso negado quando não autorizado |
| BIA-03 | Ingestão documental | Upload, hashes, deduplicação, direitos e importação bibliográfica |
| BIA-04 | Extração rastreável | Páginas, trechos, offsets e status de falha/revisão |
| BIA-05 | Busca lexical independente | Índice BM25 próprio, filtros e resultados documentados |
| BIA-06 | Anotação científica | UI/API para BIO spans, revisão e exportação de dataset |
| BIA-07 | Tokenizador e modelo próprios | Tokenizador treinável e modelo sem checkpoints importados |
| BIA-08 | Treinamento e avaliação | Jobs locais, seeds, snapshots por documento, relatórios por classe |
| BIA-09 | Inferência com evidências | Trechos/DOIs/páginas e abstinência quando faltar suporte |
| BIA-10 | Interface de pesquisador | Importar, buscar, revisar e consultar extrações |
| BIA-11 | Segurança e conformidade | Auditoria, restrições de acesso, direitos, limites de upload |
| BIA-12 | Documentação, testes e CI | Guia reproduzível, dados sintéticos, testes e gate de regressão |

## 10. Critérios de aceitação do Bloco 01

### Funcionais
- [ ] Um PDF digital **autorizado** entra e é registrado com hash, metadados e direitos.
- [ ] Texto por página pode ser inspecionado; trechos mantêm fonte e offsets.
- [ ] PDFs sem texto recuperável são encaminhados à revisão; falha não é interpretada como “sem conteúdo”.
- [ ] Busca lexical devolve resultados ordenados, com fonte verificável.
- [ ] Especialistas conseguem anotar/corrigir/aprovar rótulos.
- [ ] É possível gerar snapshot versionado e divisão por documento.
- [ ] Modelo inicia sem nenhum peso externo e treina localmente num conjunto de teste sintético.
- [ ] Pesos treinados podem ser recarregados com o mesmo tokenizador e configuração.
- [ ] Inferência informa a versão do modelo e apresenta evidências, ou se abstém.
- [ ] Interface permite um percurso completo de ingestão até revisão da saída.

### Qualidade e segurança
- [ ] Testes de parser, offsets, tokenizador round-trip, BM25, modelo, checkpoints, permissões e API.
- [ ] Rodadas repetidas com a mesma seed e ambiente fixado têm resultados reprodutíveis dentro de tolerâncias documentadas.
- [ ] Não há chamadas de inferência/embeddings a serviços externos nem download implícito de modelos.
- [ ] Treino, validação e teste isolados por documento e protegidos contra duplicatas/versões equivalentes.
- [ ] Métricas por classe: precisão, recall e F1; busca: Recall@k/MRR; conformidade: taxa de respostas com evidência verificável.
- [ ] Toda medição usa dados realmente disponíveis; **não** alegar precisão ou F1 antes de avaliar.
- [ ] Documentos restritos não são publicados no Git nem em artefatos de CI.
- [ ] CI verde em teste CPU e baseline fixado; mudanças não reduzem cobertura por atalhos.

### Gate científico
A passagem para produção exige que o grupo tripartite defina o recorte, aprove o esquema de anotação e estabeleça critérios numéricos mínimos para cada tarefa de pesquisa. Até lá, o sistema permanece como protótipo de pesquisa. Quantidade de artigos, orçamento de GPU e thresholds de qualidade **não constam como definidos** nos documentos fornecidos.

## 11. Dependências e decisões pendentes

1. **Recorte temático** compartilhado entre os três INCTs e as primeiras perguntas científicas de pesquisa.
2. **Política de uso e treinamento dos artigos:** PDFs abertos versus documentos de assinatura CAPES, limitações de licenças e armazenamento.
3. **Equipe responsável pela validação**, diretrizes de anotação e fluxo de divergências entre revisores.
4. **Infraestrutura** já disponível (CPU, GPU, RAM, disco, Linux/Windows, servidor interno).
5. **Repositório Git** da bIA e convenções de contribuição.
6. **Critérios de aceitação científica:** definição formal posterior aos primeiros dados reais e ao baseline.

O bloco pode ser iniciado tecnicamente sem recorte científico fechado, usando corpus **sintético** e módulos configuráveis. Entretanto, não será legítimo apresentar o modelo como validado cientificamente sem dados e avaliação aprovados.

## 12. Próximo bloco (após o primeiro passar os gates)

**Bloco 02 — Inteligência de relações experimentais:** extração de relações entre material-processo-condição-propriedade, normalização de unidades sob regras científicas revisadas, consolidação de resultados em tabelas estruturadas e primeiras consultas comparativas entre estudos. O avanço dependerá da qualidade dos spans e das relações aprovadas no Bloco 01.

---

### Declaração de conclusão esperada

> **Ao final do Bloco 01, a bIA terá um primeiro núcleo próprio e funcional, não um LLM científico completo:** dados estruturados com proveniência, busca rastreável, interface de revisão e primeiro modelo neural de extração escrito e treinado pelo projeto. Seus resultados serão considerados experimentais até validação pelos pesquisadores dos três INCTs.
