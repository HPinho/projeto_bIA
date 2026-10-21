# bIA — inteligência científica de materiais

Núcleo experimental da **bIA**, desenvolvido com código próprio de busca, tokenização, arquitetura neural, treinamento e inferência local, **sem APIs de IA comerciais, checkpoints pré-treinados ou download automático de pesos**.

> **Estado:** Bloco 01 (protótipo de pesquisa). A IA ainda não foi validada cientificamente. A proposta dos INCTs define construção do acervo, extração e validação tripartite antes de adoção científica.

## Executar localmente

Requisitos: Python 3.11+, Node.js 22+ (para a interface), CPU. PostgreSQL é opcional. A configuração SQLite é adequada apenas à fase local.

```bash
python -m venv .venv
# Ativar .venv conforme seu sistema operacional
pip install -e '.[ml]'
cp .env.example .env
# Exportar as variáveis de .env em seu shell, sobretudo BIA_API_KEYS; não usar tokens do exemplo fora do localhost
uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
```

No Windows/PowerShell, use `Copy-Item .env.example .env` e configure as variáveis de ambiente manualmente. O Python não carrega `.env` automaticamente. **Sem `BIA_API_KEYS`, todas as rotas científicas permanecem bloqueadas**; `/health` e `/docs` continuam públicos.

Exemplo rápido de configuração em Bash para desenvolvimento local:

```bash
export BIA_API_KEYS='{"dev-curator-CHANGE":{"actor":"curator","role":"curator"},"dev-reviewer-CHANGE":{"actor":"reviewer","role":"curator"},"dev-admin-CHANGE":{"actor":"admin","role":"admin"}}'
export BIA_DATABASE_URL='sqlite:///./data/bia.sqlite3'
export BIA_STORAGE_DIR='./data/documents'
```

Interface independente (outro terminal):

```bash
cd apps/web
npm install
npm run dev
```

Abrir http://localhost:3000, informar token local e importar **apenas PDFs autorizados**. Backend: http://localhost:8000/docs.

## Fluxo científico

1. Um curador cadastra um PDF digital permitido via `POST /v1/documents` (multipart: `file`, `title`, `rights_basis`, `training_allowed`; DOI e ano opcionais).
2. Pesquisadores consultam `GET /v1/search?q=celulose`, sem chamar provedores externos. Cada resultado inclui evidência e página.
3. Curador anota rótulos por offsets via `POST /v1/annotations`; outro curador revisa via `POST /v1/annotations/{id}/review`.
4. Quando **todas** as anotações estiverem revisadas, um curador submete o trecho (`POST /v1/passages/{id}/submit`) e **outra identidade** certifica que a anotação cobre todo o texto (`POST /v1/passages/{id}/certify`). Trechos sem entidades podem ser certificados explicitamente como exemplos negativos. Trechos em rascunho **não** servem como negativos. Uma correção posterior exige `POST /v1/passages/{id}/reopen`, retirando imediatamente o trecho de novos datasets.
5. Visualize a prontidão local via `GET /v1/curation/summary` (curadores e administradores). Somente um administrador solicita `POST /v1/models/train` com JSON, por exemplo `{"epochs": 2, "seed": 42}`. Consultar `GET /v1/models/runs/{id}`.
6. `GET /v1/extract/{passage_id}` executa o checkpoint local, retorna offsets, texto literal, versão e estado experimental. Se não houver modelo, retorna `insufficient_evidence`.

Rótulos iniciais (provisórios): `MATERIAL`, `PROCESS`, `PROPERTY`, `QUANTITY`, `UNIT`, `CONDITION`, `APPLICATION`.

## Dados certificados e avaliação

- Apenas documentos com `training_allowed=true` **e** trechos certificados entram no treino; isso evita transformar desconhecimento em rótulo negativo `O`.
- O protocolo de submissão/certificação exige duas identidades diferentes. Uma revisão de anotação individual **não** certifica o texto completo.
- Entidades BIO não são cortadas entre janelas; entidades maiores que `max_length` causam erro explícito e exigem reconfiguração. Nenhum exemplo é descartado silenciosamente.
- Splits permanecem agrupados por SHA-256 de documento; o vocabulário é ajustado **somente com documentos de treino**. Validação mede spans de entidade completos, com métricas nulas quando os dados são insuficientes.
- Cada execução registra hash SHA-256 do snapshot, das configurações do modelo, vocabulário e pesos. É um registro de procedência, não substitui revisão científica nem assinaturas digitais.

## Independência e ciência

A arquitetura neural usa operações PyTorch básicas e pesos aleatórios próprios. PyTorch e pypdf são infraestrutura computacional, **não** provedores de modelos. Os dados precisam ter autorização explícita para treinamento. Direitos de acesso institucional não devem ser confundidos com direito de treino.

Consultas usam recuperação lexical e **não são respostas geradas por LLM**. Inferência de entidades é experimental e exige verificação especializada. Metadados e literatura não são fabricados.

## Testes

```bash
pip install -e '.[test]'
pytest -q
```

Os testes geram PDFs e corpus sintéticos; não baixam modelos e não exigem GPU. A CI testa Python e compila o frontend. Não envie artigos restritos, pesos ou tokens ao repositório.

Consulte [a arquitetura e limitações](docs/ARCHITECTURE.md) para planejamento de produção e critérios científicos pendentes.

**Próxima frente:** Bloco 02 — extração de relações experimentais, após validação do Bloco 01 e definição do recorte pelos três INCTs.

## Bloco 02.1 — Relações entre materiais e experimentos

O grafo de relações é **curado**, não gerado automaticamente por IA. O painel oferece sugestões de pares próximos no mesmo trecho certificado, mas essa proximidade **não** é uma afirmação científica. Um curador propõe a relação e outra identidade precisa revisar a evidência literal antes de sua publicação.

- `GET /v1/passages/{id}/relation-candidates` — hipóteses com referência literal (curadores).
- `POST /v1/relations` — registra proposta em trecho certificado, com argumentos aprovados, direção tipada e offsets da evidência (curadores).
- `POST /v1/relations/{id}/review` — decisão de segundo revisor (`approved`/`rejected`).
- `GET /v1/passages/{id}/relations` — histórico de curadoria, inclusive relações obsoletas (curadores).
- `GET /v1/relations/search?q=...` — somente relações aprovadas e vigentes, respeitando os direitos do documento e incluindo DOI, página, offsets e evidência.

Reabrir ou recertificar um trecho invalida, para consulta, relações de certificações anteriores. Nenhuma relação é aprendida ou publicada automaticamente: o Transformer próprio existente extrai **entidades experimentais**, não relações.

Veja [protocolo, limitações e próxima etapa](docs/BLOCO_02_1_RELACOES.md). **Próximo item: BIA-02.2 — protocolo de negativos explícitos e treinamento de um classificador de relações com pesos próprios**, após definição das regras científicas pelos INCTs.
