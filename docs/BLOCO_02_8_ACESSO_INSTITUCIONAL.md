# BIA-02.8 — Autenticação e isolamento institucional

**Status:** implementação experimental para pesquisa. **Baseline inicial:**
`fe2fe26cbe821529bdfa835bf9d782632386922a` (CI verde).

## Entregas

1. **Vinculação de credenciais a identidades e institutos.** Perfil do token
   contém `actor`, `role` e lista de `institutes`. A API valida a configuração
   ao iniciar, rejeita identidade duplicada, papel/afiliação inválidos e compara
   digest SHA-256 dos tokens em tempo constante. `GET /v1/auth/me` não expõe
   segredos.
2. **Migração aditiva `0004`.** Coluna `documents.institute`, índice e versão
   registrados em transação com a política já existente do SQLite/PostgreSQL.
   Nenhum PDF, DOI ou anotação é removido. Documentos legados institucionais
   ficam com `institute=NULL` e só administradores globais podem acessá-los.
3. **Importação autorizada.** `POST /v1/documents` exige `institute` explícito
   para direitos institucionais. Curador deve possuir aquele instituto na
   credencial. Administrador global pode classificar qualquer instituto.
4. **Isolamento aplicado nas leituras e mutações.** Metadados, páginas,
   busca lexical, anotação, revisão, relações, duplicatas e resumos usam a
   mesma função de autorização para documentos. Identificadores de documentos
   de outros institutos retornam `404` quando consultados diretamente.
5. **Auditoria administrativa.** Leituras bem-sucedidas de metadados e páginas
   institucionais e documentos institucionais devolvidos na busca deixam
   eventos com identidade, recurso e data. `GET /v1/audit/events` exige admin.
6. **Classificação posterior.** `PUT /v1/documents/{id}/institution` exige
   administrador e justificativa. Aceita **apenas** documentos ainda sem
   instituto, impedindo transferência silenciosa; cada atribuição gera evento.
7. **Interface.** Mostra identidade/afiliação após verificação e exige seleção
   do instituto no upload restrito.

## Testes e critérios

- Testes Python legados preservados; testes novos de autenticação, acesso
  cruzado, negativos de upload, migração, auditoria e documentos legados.
- CI mantém build Next.js e testes Python. PostgreSQL 16 executa cenário
  de isolamento com fonte restrita real e verifica a versão `0004`.
- Nenhum teste é removido, ignorado ou relaxado para aprovação.

## Limitações de segurança

A autenticação ainda utiliza tokens estáticos definidos na configuração;
isto **não** equivale a autenticação institucional federada. Não há SSO,
OIDC, MFA, revogação imediata no servidor nem isolamento por linha no banco.
Administradores globais acessam todos os institutos por design. O uso do
checkpoint conjunto também não implementa segregação por instituto, embora
sua criação continue limitada ao administrador e exija autorização explícita
de treinamento por documento. É necessário uso de HTTPS, autenticação
institucional, revisão formal de direitos e testes independentes antes de
armazenar acervos sensíveis em produção.

## Próximo passo: BIA-02.9

Gerenciamento de identidades e tokens revogáveis; política de compartilhamento
entre institutos; isolamento de datasets, checkpoints e artefatos; auditoria
abrangente de ações de leitura e protocolo com pesquisadores dos três INCTs.
