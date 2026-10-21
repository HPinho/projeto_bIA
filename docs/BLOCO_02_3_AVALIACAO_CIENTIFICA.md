# BIA-02.3 — Avaliação científica auditável por publicação

**Estado: implementação técnica experimental.** Não existem, neste repositório, resultados comprovados em um corpus real representativo dos três INCTs. Os testes automatizados usam dados sintéticos e não medem capacidade científica.

## Escopo entregue

- A classificação de relações continua treinada localmente, do zero, com PyTorch utilizado para cálculo; nenhum checkpoint pré-treinado nem API comercial de IA é consumido.
- O conjunto supervisionado mantém apenas decisões de **aprovação** ou de **ausência explícita da relação** após certificação completa por identidades independentes, com direito de treinamento documentado.
- Separação determinística por documento, **DOI normalizado** e conteúdo idêntico de passagem: versões diferentes de um artigo e transcrições idênticas não podem migrar entre treino/validação/teste. DOI ausente não causa agrupamento artificial com outros artigos sem DOI.
- O tokenizador aprende apenas no treino. Os pesos só são atualizados com amostras do treino. Validação e teste são avaliados sem gradiente ao final, nunca entram na otimização.
- `dataset_snapshot.json` guarda um inventário estável de registros, offsets, hashes de publicação/documento/passagem, estado de curadoria, identificação de revisão e split. O SHA-256 dos bytes do snapshot integra `manifest.json`.
- `evaluation_report.json` é separado do treinamento e contém resumo por tipo de relação, por documento, macro-F1 calculado somente para tipos mensuráveis, matriz TP/TN/FP/FN, erros de classificação e contagem de documentos por split.
- O manifesto registra SHA-256 do relatório, do snapshot, do tokenizador e dos pesos. Leitura do relatório falha quando hashes não conferem. SHA-256 fornece integridade por comparação com o manifesto armazenado, não autenticação criptográfica contra adulteração de todos os arquivos.
- Apenas `admin` acessa `GET /v1/relation-models/runs/{run_id}/evaluation`. Os relatórios contêm IDs/hashes e indicadores numéricos, mas **não** publicam textos integrais ou conteúdo de artigos. O acesso exige execução concluída e diretório esperado.
- O frontend permite consultar o identificador do dataset, a contagem de publicações, F1 em validação/teste e número de erros, usando `indefinido` quando não existe suporte amostral.

## Interpretação responsável

1. Métricas de validação e teste são diferentes. A validação pode orientar investigação de erros; **não** utilizar o teste retidamente para escolher hiperparâmetros. A análise do teste é diagnóstica e final, não uma nova fonte de treinamento.
2. `precision`, `recall` e `f1` podem ser **`null`** quando o denominador não existe; isso não equivale a valor zero. `holdout_evaluable=false` sinaliza inexistência de amostras de teste. A divisão pode ficar sem validação ou teste com poucos grupos independentes, especialmente quando é preciso garantir as duas classes no treino; o sistema **não** inventa valores.
3. Escores `score_uncalibrated` não são probabilidades calibradas. Calibração, intervalo de confiança, avaliação de concordância entre anotadores e critérios de aprovação científica exigem estudos reais e desenho experimental aprovados pelos INCTs.
4. Uma F1 alta em dados sintéticos pequenos não prova utilidade nem generalização para materiais. A seleção do recorte, perguntas e tipos de evidência deve ser acordada entre INCT InovaLigno, INCT Polissacarídeos e INCT Circularidade.
5. Hash de DOI serve apenas para agrupar publicações com mesmo DOI declarado. Estudos equivalentes com DOIs diferentes e paráfrases quase idênticas podem ainda provocar vazamento; validação de duplicatas bibliográficas por curadores permanece necessária.

## Artefatos de cada treino

```text
checkpoints/relations/<run_id>/
  weights.pt
  tokenizer.json
  dataset_snapshot.json
  evaluation_report.json
  manifest.json
```

- A configuração do modelo, seed, epochs, tipos de relação, hashes e métricas estão no manifesto. O JSON é guardado localmente, fora do Git.
- Resultados por documento e lista de erros são restritos a usuários autorizados; não disponibilizar a publicação de hashes de corpora institucionais para público sem revisão jurídica.
- Após alterações de curadoria ou permissão, iniciar **novo treino** para gerar novo snapshot e versão de avaliação. Os artefatos existentes representam estado histórico, não elegibilidade atual para novos treinamentos.

## Protocolo de validação científica a iniciar com os três INCTs

| Etapa | Responsáveis | Critério antes de aceitar como evidência |
|---|---|---|
| Definir recorte comum | Representantes dos três INCTs | Área e perguntas científicas formalmente aprovadas |
| Organizar literatura | Bibliotecários/curadores de conteúdo | DOI normalizado, autorização de acesso e treino, controle de duplicatas |
| Criar amostra de referência | Especialistas no domínio | Dupla anotação independente, resolução documentada de conflitos, orientação sobre negativos explícitos |
| Congelar teste externo | Grupo tripartite | Publicações inéditas para desenvolvimento, grupos sem sobreposição de versões |
| Avaliar modelo | Equipe de computação + especialistas | Relatório por tipo, por instituto e por documento, incluindo FPs/FNs e exemplos com referência no ambiente controlado |
| Publicar resultados | Comitê científico e jurídico | Aprovação do protocolo, incertezas, limitações e política de divulgação dos dados |

**Próxima implementação proposta — BIA-02.4:** protocolo de adjudicação de anotações, duplicatas sem DOI, splits estratificados por domínio/INCT quando existir metadado confiável, e concordância entre revisores. Não declarar acurácia real até existir referência humana aprovada.
