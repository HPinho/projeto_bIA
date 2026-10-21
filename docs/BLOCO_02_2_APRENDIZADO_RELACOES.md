# BIA-02.2 — Classificador neural próprio de relações científicas

**Estado:** protótipo de pesquisa com treinamento local, não validado em artigos reais.
**Baseline anterior:** `41fd4851620bb7f1ca411dd333b127cb402f4e10` (CI verde).

## Objetivo e arquitetura

Classificar propostas *tipadas e direcionais* de relações dentro de uma passagem científica certificada. O projeto implementa um **Transformer de caracteres próprio**, pesos inicializados aleatoriamente, codificação posicional, blocos de atenção, pooling separado para contexto/entidade sujeito/entidade objeto e embedding do tipo de relação, seguido por cabeçalho binário. PyTorch fornece somente operações, autodiferenciação e otimizador; nenhum modelo, checkpoint, API de IA ou embedding externo é utilizado.

A previsão é uma **pontuação experimental NÃO calibrada**. A relação só integra o grafo científico depois de proposta e aprovação independente, pelo fluxo humano BIA-02.1. Treinar o classificador **não modifica** qualquer anotação, certificação ou relação do grafo.

## Protocolo de rotulagem

- **Positivo (1):** relação aprovada por outra identidade, baseada em entidades aprovadas de uma passagem certificada vigente, no documento especificamente autorizado para treino.
- **Negativo (0):** relação explicitamente rejeitada **porque a relação está ausente**, sinalizada pelo segundo revisor com `explicit_no_relation=true`. Rejeições genéricas **não** viram negativos (podem indicar evidência ruim, anotação errada ou incerteza).
- Propostas pendentes, pares próximos sem curadoria, relações obsoletas, trechos reabertos, documentos sem `training_allowed=true`, argumentos reprovados e revisões pela mesma identidade não entram no treino.
- Um negativo vale somente para o triplo **(sujeito, tipo de relação, objeto)** e sua evidência: não é negativo automático dos demais tipos, sentidos ou pares.
- A nova tabela aditiva `relation_negative_assessments` documenta a declaração de ausência. Não se modifica o status de relações antigas. Apenas rejeições **novas** com declaração explícita podem ser exemplos negativos.

O modelo recusa a execução caso o conjunto de treino não contenha exemplos de **ambas as classes**. Dados sintéticos dos testes verificam o mecanismo, não sua efetividade científica.

## Split, integridade e reprodutibilidade

1. Os conjuntos `train`, `val` e `test` são separados por **documento SHA-256** e mantêm no mesmo grupo documentos com **passagens textualmente idênticas**. O split depende da semente e é determinístico.
2. Se o número de grupos/classe for insuficiente, o algoritmo pode manter amostras adicionais em `train`, reportando métricas de validação como `null` quando não houver validação. **Não inventamos métricas.**
3. O vocabulário de caracteres é calculado **somente** a partir dos documentos de treinamento. Holdouts não entram na construção do vocabulário nem no ajuste de pesos; teste fica reservado para avaliação futura independente.
4. A janela textual preserva **integralmente as duas entidades** e sua direção. Pares além do `max_length` falham com erro; não desaparecem silenciosamente do treinamento.
5. O diretório único do run salva `weights.pt`, `tokenizer.json`, `dataset_snapshot.json`, `manifest.json` com hashes SHA-256, hiperparâmetros, contagens por classe, conjunto e métricas por tipo. Inferência verifica os hashes antes de carregar pesos com `weights_only=True`.
6. Documentos restritos e identificadores não são enviados a provedores de IA, e o snapshot registra hashes/IDs de proveniência, não o conteúdo de cada documento. Artefatos e textos licenciados não devem ser enviados para o GitHub público.

**Limitações:** agrupar texto idêntico não detecta paráfrases nem derivações da mesma amostra; amostras de trechos distintos do mesmo estudo devem manter ID de estudo comum numa fase futura. Métricas internas, especialmente com poucos dados, não comprovam qualidade científica. Escore via softmax não é confiança calibrada; não usar limiar para declarar verdade científica. Também não há mineração automática de novos pares fora das sugestões tipadas do BIA-02.1.

## API (Bearer)

| Rota | Papel | Função |
| --- | --- | --- |
| `POST /v1/relations/{id}/review` | curator/admin | `{"decision":"rejected","explicit_no_relation":true}` — negativo explícito; `false` (padrão) é rejeição sem rótulo para ML |
| `POST /v1/relation-models/train` | admin | `{"epochs":2,"seed":42}` — solicita treinamento local em fila |
| `GET /v1/relation-models/runs/{id}` | curator/admin | Status, métricas auditáveis ou erro |
| `GET /v1/passages/{id}/relation-predictions?run_id=<opcional>` | curator/admin | Sugestões por escore **não calibrado**, com evidência literal, sem aprovação automática |

`GET /v1/relations/search` permanece **exclusivamente humano**, sem expor previsões não revisadas. A tela da bIA permite iniciar treino (administrador), consultar execução, pedir previsões e revisar/rejeitar relações com motivos distintos. A permissão do documento e a certificação vigente são verificadas em cada consulta.

## Reproduzir

```bash
pip install -e '.[test]'
python -m pytest -q
python -m compileall -q bia_core apps tests
cd apps/web && npm install --no-audit --no-fund && npm run build
```

O `pip` e o `npm` buscam bibliotecas de infraestrutura; não há download de pesos nem uso de APIs de modelos. Para treinar com dados reais, pesquisadores precisam autorizar legalmente os documentos, aprovar entidades, certificar passagens, propor e revisar **positivos e negativos explícitos**. Sem corpus supervisionado suficiente, o endpoint retorna falha controlada no job.

## Próximo bloco: BIA-02.3

- Definição científica tripartite dos tipos de relação/negativos, critérios de evidência e base anotada.
- Avaliação **fora da amostra** (holdout por estudo/publicação), curvas PR por predicado, calibração e análise de erros junto aos INCTs.
- Versionamento/migrações de banco via Alembic, reprodutibilidade de infraestrutura e gestão de datasets autorizados para implantação institucional.
- Evoluir extração de pares candidatos/condições experimentais sem tratá-los como conclusões validadas.
