'use client';
import { FormEvent, useState } from 'react';

const BASE = process.env.NEXT_PUBLIC_BIA_API_URL || 'http://localhost:8000';
type Hit = { document_id: string; title: string; doi: string | null; passage_id: string; page: number; score: number; quote: string };
type ReviewState = { status: string; submitted_by: string | null; certified_by: string | null; approved_annotations: number; pending_annotations: number; rejected_annotations: number; unresolved_disputes: number };
type Passage = { id: string; document_id: string; page: number; text: string; review: ReviewState };
type Annotation = { id: string; label: string; start: number; end: number; quote: string; status: string };
type CandidateRelation = { subject_annotation_id: string; object_annotation_id: string; relation_type: string; evidence_start: number; evidence_end: number; evidence_quote: string; status: string };
type PredictedRelation = CandidateRelation & { model_score_uncalibrated: number; scientifically_validated: false };
type CuratedRelation = { id: string; relation_type: string; status: string; verified: boolean; certification_current: boolean; evidence_quote: string; proposed_by: string; reviewed_by: string | null };
type DisputeAlternative = { actor: string; label: string; start: number; end: number; quote: string };
type Dispute = { id: string; status: string; first: DisputeAlternative; second: DisputeAlternative | null; resolution: string | null; reason: string | null; adjudicated_by: string | null; annotation_id: string | null };
type Duplicate = { document_id: string; title: string; doi: string | null; year: number | null; signals: string[]; verdict: string; reviewed_by: string | null };
type AgreementTask = { id: string; passage_id: string; quote: string; start: number; end: number; raters: string[]; complete: boolean; votes_received: number; needs_adjudication: boolean; adjudicated: boolean; votes?: { a: string; b: string }; final_label?: string | null };
type AgreementMetric = { rater_a: string; rater_b: string; tasks: number; complete: number; agreement_rate: number | null; cohen_kappa: number | null; kappa_defined: boolean };
type CorpusRecord = { id: string; title: string; rights_basis: string; status: string; training_allowed: boolean; certified_passages: number; readiness_issues: string[] };
type CorpusOverview = { documents: number; restricted_documents: number; training_authorized_ready_documents: number; training_authorized_certified_passages: number; legal_holds: number; retention_review_due: number; scientifically_validated: boolean };
type VerifiedRelation = { id: string; title: string; doi: string | null; page: number; relation_type: string; subject: { quote: string }; object: { quote: string }; evidence: { quote: string }; review: { proposed_by: string; reviewed_by: string } };

export default function Research() {
  const [token, setToken] = useState('');
  const [corpusOverview, setCorpusOverview] = useState<CorpusOverview | null>(null);
  const [corpusRecords, setCorpusRecords] = useState<CorpusRecord[]>([]);
  const [query, setQuery] = useState('lignocelulose');
  const [hits, setHits] = useState<Hit[]>([]);
  const [passages, setPassages] = useState<Passage[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [disputes, setDisputes] = useState<Dispute[]>([]);
  const [duplicateCandidates, setDuplicateCandidates] = useState<Duplicate[]>([]);
  const [bibliographyReason, setBibliographyReason] = useState('Documentos comparados com evidências bibliográficas originais');
  const [adjudicationReason, setAdjudicationReason] = useState('Texto original e offsets revisados pelos três pesquisadores');
  const [reviewState, setReviewState] = useState<ReviewState | null>(null);
  const [selected, setSelected] = useState('');
  const [message, setMessage] = useState('');
  const [modelOutput, setModelOutput] = useState('');
  const [candidates, setCandidates] = useState<CandidateRelation[]>([]);
  const [predictions, setPredictions] = useState<PredictedRelation[]>([]);
  const [relationRunId, setRelationRunId] = useState('');
  const [relationRunStatus, setRelationRunStatus] = useState('');
  const [relationEvaluation, setRelationEvaluation] = useState<{ dataset_sha256: string; scientifically_validated: boolean; documents: Record<string, number>; validation: { overall: { examples: number; f1: number | null }; errors: { kind: string; relation_type: string }[] }; test: { overall: { examples: number; f1: number | null }; errors: { kind: string; relation_type: string }[] } } | null>(null);
  const [curatedRelations, setCuratedRelations] = useState<CuratedRelation[]>([]);
  const [relationResults, setRelationResults] = useState<VerifiedRelation[]>([]);
  const [relationQuery, setRelationQuery] = useState('celulose');
  const [agreementTasks, setAgreementTasks] = useState<AgreementTask[]>([]);
  const [raterA, setRaterA] = useState('alice');
  const [raterB, setRaterB] = useState('bruno');
  const [agreementLabel, setAgreementLabel] = useState('MATERIAL');
  const [agreementReason, setAgreementReason] = useState('Decisão conferida com o documento original e seus offsets');
  const [agreementMetrics, setAgreementMetrics] = useState<AgreementMetric[] | null>(null);
  const [agreementExport, setAgreementExport] = useState('');
  const [label, setLabel] = useState('MATERIAL');
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(1);
  const [title, setTitle] = useState('');
  const [rights, setRights] = useState('open_access');
  const [trainingAllowed, setTrainingAllowed] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  async function api(path: string, init: RequestInit = {}) {
    const headers = new Headers(init.headers);
    headers.set('Authorization', `Bearer ${token}`);
    const response = await fetch(BASE + path, { ...init, headers, cache: 'no-store' });
    const json = await response.json();
    if (!response.ok) throw new Error(JSON.stringify(json.detail || json));
    return json;
  }
  async function exec(fn: () => Promise<void>) {
    setMessage('');
    try { await fn(); } catch (e) { setMessage((e as Error).message); }
  }
  async function loadCorpus() {
    await exec(async () => {
      setCorpusOverview(await api('/v1/corpus/readiness'));
      const manifest = await api('/v1/corpus/manifest?offset=0&limit=100');
      setCorpusRecords(manifest.items);
      setMessage('Inventário interno recuperado. Os dados ainda exigem validação científica.');
    });
  }
  async function search(e: FormEvent) {
    e.preventDefault();
    await exec(async () => {
      const data = await api('/v1/search?q=' + encodeURIComponent(query));
      setHits(data.results); setMessage(data.results.length ? `Encontrados ${data.results.length} trechos.` : 'Evidência insuficiente para esta busca.');
    });
  }
  async function loadPassages(documentId: string) {
    await exec(async () => {
      const result: Passage[] = await api(`/v1/documents/${documentId}/passages`);
      setPassages(result); setSelected(result[0]?.id || '');
      setCandidates([]); setCuratedRelations([]); setPredictions([]); setModelOutput(''); setDisputes([]); setDuplicateCandidates([]);
      if (result[0]) { setAnnotations(await api('/v1/annotations/' + result[0].id)); setReviewState(result[0].review); }
      else { setAnnotations([]); setReviewState(null); }
    });
  }
  async function choosePassage(id: string) {
    await exec(async () => {
      setSelected(id); setCandidates([]); setCuratedRelations([]); setPredictions([]); setModelOutput(''); setDisputes([]);
      setAnnotations(await api('/v1/annotations/' + id));
      setReviewState(await api('/v1/passages/' + id + '/review'));
    });
  }
  async function changeReview(action: 'submit' | 'certify' | 'reopen') {
    await exec(async () => {
      setReviewState(await api(`/v1/passages/${selected}/${action}`, { method: 'POST' }));
      setCandidates([]); setCuratedRelations([]); setPredictions([]);
      setMessage(`Trecho: ${action} concluído. Atualize o estado antes de treinar.`);
    });
  }
  async function upload(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    await exec(async () => {
      const data = new FormData();
      data.append('file', file); data.append('title', title);
      data.append('rights_basis', rights); data.append('training_allowed', String(trainingAllowed));
      const result = await api('/v1/documents', { method: 'POST', body: data });
      setMessage(`Documento ${result.id} registrado (${result.status}).`);
      await loadPassages(result.id);
    });
  }
  async function annotate(e: FormEvent) {
    e.preventDefault();
    await exec(async () => {
      await api('/v1/annotations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ passage_id: selected, label, start, end }) });
      setAnnotations(await api('/v1/annotations/' + selected));
      setReviewState(await api('/v1/passages/' + selected + '/review'));
    });
  }
  async function review(id: string, decision: string) {
    await exec(async () => {
      await api(`/v1/annotations/${id}/review`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ decision }) });
      setAnnotations(await api('/v1/annotations/' + selected));
      setReviewState(await api('/v1/passages/' + selected + '/review'));
    });
  }
  async function refreshDisputes() {
    if (!selected) return;
    const results = await api(`/v1/passages/${selected}/adjudications`);
    setDisputes(results.cases);
    setReviewState(await api(`/v1/passages/${selected}/review`));
    setAnnotations(await api(`/v1/annotations/${selected}`));
  }
  async function createDispute() {
    await exec(async () => {
      await api('/v1/adjudications', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ passage_id: selected, alternative: { label, start, end } }) });
      await refreshDisputes(); setMessage('Primeira opinião registrada; outro pesquisador deve enviar a segunda.');
    });
  }
  async function secondDispute(id: string) {
    await exec(async () => {
      await api(`/v1/adjudications/${id}/second`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label, start, end }) });
      await refreshDisputes(); setMessage('Segunda opinião registrada. Uma terceira identidade deverá adjudicar.');
    });
  }
  async function resolveDispute(id: string, resolution: 'first' | 'second' | 'neither') {
    await exec(async () => {
      await api(`/v1/adjudications/${id}/resolve`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ resolution, reason: adjudicationReason }) });
      await refreshDisputes(); setMessage('Divergência adjudicada e registrada em auditoria.');
    });
  }
  async function refreshAgreement() {
    if (!selected) return;
    const result = await api('/v1/agreement/tasks?passage_id=' + encodeURIComponent(selected));
    setAgreementTasks(result.tasks);
  }
  async function createAgreementTask() {
    await exec(async () => {
      await api('/v1/agreement/tasks', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ passage_id: selected, start, end, rater_a: raterA, rater_b: raterB }) });
      await refreshAgreement(); setMessage('Intervalo definido antes dos votos; cada revisor deve entrar com seu próprio token.');
    });
  }
  async function castAgreementVote(id: string) {
    await exec(async () => {
      await api(`/v1/agreement/tasks/${id}/vote`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: agreementLabel }) });
      await refreshAgreement(); setMessage('Voto registrado sem revelar a opção ao outro revisor.');
    });
  }
  async function adjudicateAgreement(id: string) {
    await exec(async () => {
      await api(`/v1/agreement/tasks/${id}/adjudicate`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: agreementLabel, reason: agreementReason }) });
      await refreshAgreement(); setMessage('Decisão da terceira identidade registrada sem substituir os votos originais.');
    });
  }
  async function loadAgreementMetrics() {
    await exec(async () => {
      const report = await api('/v1/agreement/metrics');
      setAgreementMetrics(report.by_fixed_rater_pair);
      setMessage(`${report.tasks} amostras pré-definidas, ${report.pending_adjudications} discordâncias pendentes.`);
    });
  }
  async function inspectGoldExport() {
    await exec(async () => {
      const snapshot = await api('/v1/datasets/gold/export');
      setAgreementExport(`Dataset ${snapshot.sha256} · ${snapshot.passages} trechos autorizados · ${snapshot.publications} publicações agrupadas. Uso interno; sem redistribuição.`);
      setMessage('Snapshot recalculado e identificado por SHA-256. A exportação íntegra está disponível no endpoint restrito a administradores.');
    });
  }
  async function inspectDuplicates() {
    await exec(async () => {
      const docId = passages[0]?.document_id;
      if (!docId) throw Error('Selecione um documento antes de pesquisar duplicatas.');
      const result = await api(`/v1/documents/${docId}/duplicates`);
      setDuplicateCandidates(result.candidates);
      setMessage(result.candidates.length ? 'Possíveis duplicatas: nenhum PDF foi mesclado.' : 'Nenhum possível par bibliográfico localizado.');
    });
  }
  async function reviewDuplicate(id: string, verdict: 'same_publication' | 'distinct_publication') {
    await exec(async () => {
      await api('/v1/bibliography/review', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ left_document_id: passages[0].document_id, right_document_id: id, verdict, reason: bibliographyReason }) });
      await inspectDuplicates(); setMessage('Parecer bibliográfico registrado; fontes e direitos preservados.');
    });
  }
  async function loadRelations() {
    await exec(async () => {
      const [suggestions, existing] = await Promise.all([
        api(`/v1/passages/${selected}/relation-candidates`),
        api(`/v1/passages/${selected}/relations`),
      ]);
      setCandidates(suggestions.candidates);
      setCuratedRelations(existing.relations);
      setMessage('Sugestões são hipóteses; nenhuma relação é publicada sem revisão independente.');
    });
  }
  async function propose(candidate: CandidateRelation) {
    await exec(async () => {
      await api('/v1/relations', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ passage_id: selected, ...candidate }) });
      const existing = await api(`/v1/passages/${selected}/relations`);
      setCuratedRelations(existing.relations);
      setMessage('Relação proposta. Outro curador precisa revisar a evidência original.');
    });
  }
  async function reviewRelation(relationId: string, decision: 'approved' | 'rejected', explicitNoRelation = false) {
    await exec(async () => {
      await api(`/v1/relations/${relationId}/review`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision, explicit_no_relation: explicitNoRelation }) });
      const existing = await api(`/v1/passages/${selected}/relations`);
      setCuratedRelations(existing.relations);
      setMessage('Revisão registrada. Apenas relações aprovadas com certificação vigente são consultáveis.');
    });
  }
  async function trainRelationModel() {
    await exec(async () => {
      const run = await api('/v1/relation-models/train', { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ epochs: 2, seed: 42 }) });
      setRelationRunId(run.id); setRelationRunStatus(run.status);
      setMessage('Treinamento local solicitado. Somente administrador pode iniciar. Consulte o status.');
    });
  }
  async function relationTrainingStatus() {
    await exec(async () => {
      if (!relationRunId) throw Error('Informe o identificador do treinamento.');
      const run = await api('/v1/relation-models/runs/' + encodeURIComponent(relationRunId));
      setRelationRunStatus(run.status);
      setMessage(JSON.stringify({ status: run.status, metrics: run.metrics, error: run.error }, null, 2));
    });
  }
  async function inspectRelationEvaluation() {
    await exec(async () => {
      if (!relationRunId) throw Error('Informe o ID do treinamento concluído.');
      const report = await api('/v1/relation-models/runs/' + encodeURIComponent(relationRunId) + '/evaluation');
      setRelationEvaluation(report);
      setMessage('Avaliação experimental recuperada com verificação de integridade. Acesso somente de administrador.');
    });
  }
  async function predictRelations() {
    await exec(async () => {
      const query = relationRunId ? `?run_id=${encodeURIComponent(relationRunId)}` : '';
      const response = await api(`/v1/passages/${selected}/relation-predictions${query}`);
      setPredictions(response.candidates);
      setMessage(response.warning || response.reason || 'Previsões experimentais não verificadas.');
    });
  }
  async function searchRelations(e: FormEvent) {
    e.preventDefault();
    await exec(async () => {
      const found = await api('/v1/relations/search?q=' + encodeURIComponent(relationQuery));
      setRelationResults(found.results);
      setMessage(found.results.length ? `${found.results.length} relações revisadas com evidência.` : 'Nenhuma relação revisada localizada.');
    });
  }
  return <main>
    <header><span className="badge">Bloco 02.5 · Concordância científica e datasets auditáveis</span><h1>bIA</h1><p className="muted">Inteligência de materiais com fontes verificáveis · modelo próprio · processamento local</p></header>
    <section className="card stack"><h2>Conexão com o laboratório</h2><label>Token de acesso (somente memória desta aba)</label><input type="password" value={token} autoComplete="off" onChange={e => setToken(e.target.value)} placeholder="Bearer token configurado na API" /><small>Use tokens diferentes para criar e revisar anotações. Não utilize este protótipo publicamente sem autenticação institucional.</small></section>
    <section className="card stack"><h2>Inventário científico e governança · BIA-02.7</h2>
      <p className="muted">Visão administrativa restrita: fontes, autorização de treinamento e revisão de retenção. Não é uma certificação científica nem autoriza exclusões automáticas.</p>
      <button type="button" disabled={!token} onClick={loadCorpus}>Consultar inventário autorizado</button>
      {corpusOverview && <p>Documentos: {corpusOverview.documents} · elegíveis para treino: {corpusOverview.training_authorized_ready_documents} · trechos certificados autorizados: {corpusOverview.training_authorized_certified_passages} · acesso restrito: {corpusOverview.restricted_documents} · guarda legal: {corpusOverview.legal_holds} · retenções a revisar: {corpusOverview.retention_review_due}</p>}
      {corpusRecords.map(doc => <article className="result" key={doc.id}><strong>{doc.title}</strong>
        <p className="muted">{doc.status} · {doc.rights_basis} · trechos certificados: {doc.certified_passages} · treinamento: {doc.training_allowed ? 'autorizado' : 'não autorizado'}</p>
        {doc.readiness_issues.length > 0 && <small>Pendências: {doc.readiness_issues.join(', ')}</small>}
      </article>)}
      {corpusOverview && <small>São exibidos até 100 documentos por consulta. Revogação de direitos e guarda legal somente pela API administrativa autenticada.</small>}
    </section>
    <section className="card"><h2>Importar documento científico autorizado</h2><form className="stack" onSubmit={upload}>
      <input required placeholder="Título do trabalho" value={title} onChange={e => setTitle(e.target.value)} />
      <select value={rights} onChange={e => setRights(e.target.value)}><option value="open_access">Acesso aberto (com direitos verificados)</option><option value="author_permission">Permissão do autor</option><option value="institutional_permission">Permissão institucional (acesso restrito)</option></select>
      <label><input type="checkbox" checked={trainingAllowed} onChange={e => setTrainingAllowed(e.target.checked)} /> Permissão específica confirmada para treinamento local</label>
      <input type="file" accept="application/pdf,.pdf" required onChange={e => setFile(e.target.files?.[0] || null)} />
      <button disabled={!token || !file || !title}>Importar PDF</button>
    </form></section>
    <section className="card"><h2>Busca com evidências</h2><form className="row" onSubmit={search}><input value={query} onChange={e => setQuery(e.target.value)} placeholder="material, propriedade, processo..." style={{ flex: 1 }} /><button disabled={!token || query.length < 2}>Pesquisar</button></form>
      {hits.map(hit => <article className="result" key={hit.passage_id}><strong>{hit.title}</strong><p className="muted">Página {hit.page} · DOI {hit.doi || 'não informado'} · pontuação BM25 {hit.score.toFixed(3)}</p><pre>{hit.quote}</pre><button onClick={() => loadPassages(hit.document_id)}>Inspecionar e anotar</button></article>)}
    </section>
    {!!passages.length && <section className="card stack"><h2>Anotação e revisão científica</h2><select value={selected} onChange={e => choosePassage(e.target.value)}>{passages.map(p => <option key={p.id} value={p.id}>Página {p.page}</option>)}</select>
      <p className="muted">Revisão integral: <strong>{reviewState?.status || 'draft'}</strong> · aprovadas: {reviewState?.approved_annotations ?? 0} · pendentes: {reviewState?.pending_annotations ?? 0} · divergências: {reviewState?.unresolved_disputes ?? 0}. Um segundo pesquisador deve certificar inclusive trechos sem entidades.</p>
      <div className="row">
        {reviewState?.status === 'draft' && <button type="button" onClick={() => changeReview('submit')}>Submeter trecho para certificação</button>}
        {reviewState?.status === 'submitted' && <button type="button" onClick={() => changeReview('certify')}>Certificar como segundo revisor</button>}
        {(reviewState?.status === 'submitted' || reviewState?.status === 'certified') && <button type="button" onClick={() => changeReview('reopen')}>Reabrir trecho</button>}
      </div>
      <pre>{passages.find(p => p.id === selected)?.text}</pre><small>Offsets são posições de caracteres do texto extraído acima, começando em zero.</small>
      <form className="row" onSubmit={annotate}><select value={label} onChange={e => setLabel(e.target.value)}>{['MATERIAL', 'PROCESS', 'PROPERTY', 'QUANTITY', 'UNIT', 'CONDITION', 'APPLICATION'].map(v => <option key={v}>{v}</option>)}</select>
        <label>Início <input type="number" min={0} value={start} onChange={e => setStart(Number(e.target.value))} /></label><label>Fim <input type="number" min={1} value={end} onChange={e => setEnd(Number(e.target.value))} /></label><button disabled={!token || reviewState?.status !== 'draft'}>Anotar</button></form>
      {annotations.map(a => <div key={a.id} className="result"><strong>{a.label}</strong> — “{a.quote}” ({a.status}) {a.status === 'pending' && <><button onClick={() => review(a.id, 'approved')}>Aprovar</button> <button onClick={() => review(a.id, 'rejected')}>Rejeitar</button></>}</div>)}
      <section className="subpanel stack"><h3>Adjudicação de divergências (três pesquisadores)</h3>
        <p className="muted">Os campos rótulo, início e fim acima definem a opinião enviada por cada pesquisador. A certificação do trecho permanece bloqueada enquanto existir divergência em aberto. Uma terceira identidade escolhe a primeira opinião, a segunda ou nenhuma.</p>
        <div className="row"><button type="button" disabled={!selected || reviewState?.status !== 'draft'} onClick={createDispute}>Registrar primeira opinião</button>
          <button type="button" onClick={() => exec(refreshDisputes)}>Atualizar divergências</button></div>
        <label>Justificativa da adjudicação <input value={adjudicationReason} onChange={e => setAdjudicationReason(e.target.value)} /></label>
        {disputes.map(d => <article className="result" key={d.id}>
          <strong>{d.status}</strong> · primeira: {d.first.actor}, {d.first.label} “{d.first.quote}”
          <p>Segunda: {d.second ? `${d.second.actor}, ${d.second.label} “${d.second.quote}”` : 'aguardando identidade independente'}</p>
          {d.status === 'awaiting_second' && <button type="button" onClick={() => secondDispute(d.id)}>Enviar segunda opinião com os campos acima</button>}
          {d.status === 'awaiting_adjudication' && <div className="row"><button type="button" onClick={() => resolveDispute(d.id, 'first')}>Escolher primeira</button>
            <button type="button" onClick={() => resolveDispute(d.id, 'second')}>Escolher segunda</button>
            <button type="button" onClick={() => resolveDispute(d.id, 'neither')}>Nenhuma sustentada</button></div>}
          {d.status === 'resolved' && <small>Decisão: {d.resolution} · por {d.adjudicated_by} · motivo: {d.reason}</small>}
        </article>)}
      </section>
      <section className="subpanel stack"><h3>Concordância independente · BIA-02.5</h3>
        <p className="muted">Defina um intervalo com os campos Início/Fim acima <strong>antes</strong> da votação. As duas identidades não veem os votos até ambos terem respondido. Rótulo NONE indica ausência de entidade somente neste intervalo. Discordâncias exigem uma terceira identidade.</p>
        <div className="row"><label>Revisor A <input value={raterA} onChange={e => setRaterA(e.target.value)} /></label>
          <label>Revisor B <input value={raterB} onChange={e => setRaterB(e.target.value)} /></label>
          <button type="button" disabled={!selected || !token} onClick={createAgreementTask}>Definir amostra cega</button>
          <button type="button" onClick={() => exec(refreshAgreement)}>Atualizar amostras</button></div>
        <label>Rótulo da avaliação <select value={agreementLabel} onChange={e => setAgreementLabel(e.target.value)}>
          {['MATERIAL','PROCESS','PROPERTY','QUANTITY','UNIT','CONDITION','APPLICATION','NONE'].map(v => <option key={v}>{v}</option>)}
        </select></label>
        <label>Justificativa da terceira pessoa <input value={agreementReason} onChange={e => setAgreementReason(e.target.value)} /></label>
        {agreementTasks.map(t => <article className="result" key={t.id}><strong>“{t.quote}”</strong>
          <p className="muted">Offsets {t.start}–{t.end} · revisores {t.raters.join(' / ')} · votos {t.votes_received}/2</p>
          {t.votes && <p>Opiniões: {t.votes.a} / {t.votes.b} · rótulo final: {t.final_label || 'aguardando adjudicação'}</p>}
          {!t.complete && <button type="button" onClick={() => castAgreementVote(t.id)}>Enviar voto independente</button>}
          {t.needs_adjudication && <button type="button" onClick={() => adjudicateAgreement(t.id)}>Adjudicar como terceira pessoa</button>}
        </article>)}
        <div className="row"><button type="button" onClick={loadAgreementMetrics}>Concordância por dupla (admin)</button>
          <button type="button" onClick={inspectGoldExport}>Conferir snapshot gold (admin)</button></div>
        {agreementMetrics && <div>{agreementMetrics.map(m => <p key={m.rater_a + m.rater_b}>
          {m.rater_a} / {m.rater_b}: {m.complete}/{m.tasks} avaliações completas · concordância {m.agreement_rate?.toFixed(3) ?? 'indefinida'} · Cohen κ {m.kappa_defined ? m.cohen_kappa?.toFixed(3) : 'indefinido'}
        </p>)}</div>}
        {agreementExport && <p className="muted">{agreementExport}</p>}
        <small>Amostras selecionadas manualmente não representam automaticamente todo o acervo. O κ só é calculado por dupla fixa de avaliadores.</small>
      </section>
      <section className="subpanel stack"><h3>Duplicatas bibliográficas</h3>
        <p className="muted">Mesmo DOI ou título normalizado com ano igual sugerem revisão, não fusão automática. O revisor deve ser diferente do responsável pela ingestão. Publicações confirmadas como equivalentes ficam juntas nas divisões de treinamento e teste.</p>
        <button type="button" onClick={inspectDuplicates}>Inspecionar possíveis duplicatas deste documento</button>
        <label>Fundamentação da decisão <input value={bibliographyReason} onChange={e => setBibliographyReason(e.target.value)} /></label>
        {duplicateCandidates.map(d => <article className="result" key={d.document_id}>
          <strong>{d.title}</strong><p className="muted">DOI {d.doi || 'não informado'} · sinais: {d.signals.join(', ')} · {d.verdict}</p>
          {d.verdict === 'unreviewed' && <div className="row"><button type="button" onClick={() => reviewDuplicate(d.document_id, 'same_publication')}>Mesma publicação</button>
            <button type="button" onClick={() => reviewDuplicate(d.document_id, 'distinct_publication')}>Publicações diferentes</button></div>}
        </article>)}
      </section>
      <section className="subpanel stack"><h3>Relações entre materiais, processos e propriedades</h3>
        <p className="muted">Pares próximos no texto são <strong>somente sugestões</strong>. A relação proposta precisa incluir os dois argumentos na evidência literal e ser aprovada por outra identidade. Reabrir o trecho revoga a validade das relações anteriores.</p>
        <button type="button" disabled={!selected || !token || reviewState?.status !== 'certified'} onClick={loadRelations}>Carregar sugestões e revisões</button>
        {candidates.map((c, index) => <article className="result" key={`${c.subject_annotation_id}:${c.object_annotation_id}:${c.relation_type}:${index}`}>
          <strong>{c.relation_type}</strong><p>{c.evidence_quote}</p>
          <button type="button" disabled={reviewState?.status !== 'certified'} onClick={() => propose(c)}>Propor esta relação</button>
        </article>)}
        {curatedRelations.map(r => <article className="result" key={r.id}>
          <strong>{r.relation_type}</strong> · <span>{r.status}</span> · {r.verified ? 'revisada e vigente' : r.certification_current ? 'não publicada' : 'certificação obsoleta'}
          <p>{r.evidence_quote}</p><small>Proponente: {r.proposed_by} · Revisor: {r.reviewed_by || 'pendente'}</small>
          {r.status === 'pending' && r.certification_current && <div className="row"><button type="button" onClick={() => reviewRelation(r.id, 'approved')}>Aprovar relação</button><button type="button" onClick={() => reviewRelation(r.id, 'rejected')}>Rejeitar (sem exemplo negativo)</button><button type="button" onClick={() => reviewRelation(r.id, 'rejected', true)}>Confirmar ausência da relação (negativo para treino)</button></div>}
        </article>)}
        <div className="stack"><h3>Modelo neural de relações (experimental)</h3>
          <p className="muted">Treinado apenas com revisões atuais autorizadas, positivos aprovados e negativos de ausência declarada. Escores não são probabilidades calibradas e jamais aprovam relações automaticamente.</p>
          <div className="row"><button type="button" onClick={trainRelationModel}>Treinar modelo local (admin)</button>
            <input value={relationRunId} placeholder="ID do treinamento (opcional)" onChange={e => setRelationRunId(e.target.value)} />
            <button type="button" onClick={relationTrainingStatus}>Consultar treino</button>
            <button type="button" onClick={inspectRelationEvaluation}>Avaliação (admin)</button>
            <button type="button" disabled={reviewState?.status !== 'certified'} onClick={predictRelations}>Ordenar sugestões com IA</button></div>
          {relationRunStatus && <p className="muted">Execução: {relationRunStatus}</p>}
          {relationEvaluation && <article className="result">
            <strong>Avaliação não calibrada e sem validação científica externa</strong>
            <p className="muted">Dataset SHA-256: {relationEvaluation.dataset_sha256}</p>
            <p>Publicações por conjunto: treino {relationEvaluation.documents.train} · validação {relationEvaluation.documents.validation} · teste {relationEvaluation.documents.test}</p>
            <p>Validação: {relationEvaluation.validation.overall.examples} exemplos, F1 {relationEvaluation.validation.overall.f1?.toFixed(3) ?? 'indefinido'} · erros {relationEvaluation.validation.errors.length}</p>
            <p>Teste isolado: {relationEvaluation.test.overall.examples} exemplos, F1 {relationEvaluation.test.overall.f1?.toFixed(3) ?? 'indefinido'} · erros {relationEvaluation.test.errors.length}</p>
            <small>Ausência de dados suficientes significa métrica indefinida, não desempenho zero. Examinar erros não substitui avaliação externa dos INCTs.</small>
          </article>}
          {predictions.map((r, idx) => <article className="result" key={`${r.relation_type}-${idx}`}>
            <strong>{r.relation_type}</strong> · escore não calibrado: {r.model_score_uncalibrated.toFixed(3)}
            <p>{r.evidence_quote}</p><small>Não verificada · requer revisão humana</small>
            <button type="button" onClick={() => propose(r)}>Propor para revisão</button>
          </article>)}
        </div>
      </section>
      <button onClick={() => exec(async () => { const r = await api('/v1/extract/' + selected); setModelOutput(JSON.stringify(r, null, 2)); })}>Extrair com o modelo local</button>{modelOutput && <pre>{modelOutput}</pre>}
    </section>}
    <section className="card stack"><h2>Grafo de relações revisadas</h2>
      <p className="muted">Busca somente relações propostas e aprovadas por pessoas distintas, ligadas a evidência original e certificação vigente. Não substitui verificação científica independente.</p>
      <form className="row" onSubmit={searchRelations}><input value={relationQuery} onChange={e => setRelationQuery(e.target.value)} placeholder="Material ou propriedade" style={{ flex: 1 }} /><button disabled={!token || relationQuery.length < 2}>Buscar relações</button></form>
      {relationResults.map(r => <article className="result" key={r.id}><strong>{r.subject.quote} → {r.relation_type} → {r.object.quote}</strong>
        <p className="muted">{r.title} · Página {r.page} · DOI {r.doi || 'não informado'}</p>
        <pre>{r.evidence.quote}</pre><small>Revisão: {r.review.proposed_by} / {r.review.reviewed_by}</small>
      </article>)}
    </section>
    {message && <section className="card" role="status"><p>{message}</p></section>}
    <footer className="muted">Resultados de IA são experimentais e exigem validação de pesquisadores. Não são descobertas científicas confirmadas.</footer>
  </main>;
}
