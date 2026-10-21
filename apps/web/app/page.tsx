'use client';
import { FormEvent, useState } from 'react';

const BASE = process.env.NEXT_PUBLIC_BIA_API_URL || 'http://localhost:8000';
type Hit = { document_id: string; title: string; doi: string | null; passage_id: string; page: number; score: number; quote: string };
type ReviewState = { status: string; submitted_by: string | null; certified_by: string | null; approved_annotations: number; pending_annotations: number; rejected_annotations: number };
type Passage = { id: string; page: number; text: string; review: ReviewState };
type Annotation = { id: string; label: string; start: number; end: number; quote: string; status: string };

export default function Research() {
  const [token, setToken] = useState('');
  const [query, setQuery] = useState('lignocelulose');
  const [hits, setHits] = useState<Hit[]>([]);
  const [passages, setPassages] = useState<Passage[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [reviewState, setReviewState] = useState<ReviewState | null>(null);
  const [selected, setSelected] = useState('');
  const [message, setMessage] = useState('');
  const [modelOutput, setModelOutput] = useState('');
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
      if (result[0]) { setAnnotations(await api('/v1/annotations/' + result[0].id)); setReviewState(result[0].review); }
      else { setAnnotations([]); setReviewState(null); }
    });
  }
  async function choosePassage(id: string) {
    await exec(async () => { setSelected(id); setAnnotations(await api('/v1/annotations/' + id)); setReviewState(await api('/v1/passages/' + id + '/review')); });
  }
  async function changeReview(action: 'submit' | 'certify' | 'reopen') {
    await exec(async () => {
      setReviewState(await api(`/v1/passages/${selected}/${action}`, { method: 'POST' }));
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
  return <main>
    <header><span className="badge">Bloco 01 · Protótipo científico</span><h1>bIA</h1><p className="muted">Inteligência de materiais com fontes verificáveis · modelo próprio · processamento local</p></header>
    <section className="card stack"><h2>Conexão com o laboratório</h2><label>Token de acesso (somente memória desta aba)</label><input type="password" value={token} autoComplete="off" onChange={e => setToken(e.target.value)} placeholder="Bearer token configurado na API" /><small>Use tokens diferentes para criar e revisar anotações. Não utilize este protótipo publicamente sem autenticação institucional.</small></section>
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
      <p className="muted">Revisão integral: <strong>{reviewState?.status || 'draft'}</strong> · aprovadas: {reviewState?.approved_annotations ?? 0} · pendentes: {reviewState?.pending_annotations ?? 0}. Um segundo pesquisador deve certificar inclusive trechos sem entidades.</p>
      <div className="row">
        {reviewState?.status === 'draft' && <button type="button" onClick={() => changeReview('submit')}>Submeter trecho para certificação</button>}
        {reviewState?.status === 'submitted' && <button type="button" onClick={() => changeReview('certify')}>Certificar como segundo revisor</button>}
        {(reviewState?.status === 'submitted' || reviewState?.status === 'certified') && <button type="button" onClick={() => changeReview('reopen')}>Reabrir trecho</button>}
      </div>
      <pre>{passages.find(p => p.id === selected)?.text}</pre><small>Offsets são posições de caracteres do texto extraído acima, começando em zero.</small>
      <form className="row" onSubmit={annotate}><select value={label} onChange={e => setLabel(e.target.value)}>{['MATERIAL', 'PROCESS', 'PROPERTY', 'QUANTITY', 'UNIT', 'CONDITION', 'APPLICATION'].map(v => <option key={v}>{v}</option>)}</select>
        <label>Início <input type="number" min={0} value={start} onChange={e => setStart(Number(e.target.value))} /></label><label>Fim <input type="number" min={1} value={end} onChange={e => setEnd(Number(e.target.value))} /></label><button disabled={!token || reviewState?.status !== 'draft'}>Anotar</button></form>
      {annotations.map(a => <div key={a.id} className="result"><strong>{a.label}</strong> — “{a.quote}” ({a.status}) {a.status === 'pending' && <><button onClick={() => review(a.id, 'approved')}>Aprovar</button> <button onClick={() => review(a.id, 'rejected')}>Rejeitar</button></>}</div>)}
      <button onClick={() => exec(async () => { const r = await api('/v1/extract/' + selected); setModelOutput(JSON.stringify(r, null, 2)); })}>Extrair com o modelo local</button>{modelOutput && <pre>{modelOutput}</pre>}
    </section>}
    {message && <section className="card" role="status"><p>{message}</p></section>}
    <footer className="muted">Resultados de IA são experimentais e exigem validação de pesquisadores. Não são descobertas científicas confirmadas.</footer>
  </main>;
}
