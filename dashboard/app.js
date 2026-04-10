const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const dropHint = document.getElementById('drop-hint');

const metaSection = document.getElementById('meta');
const kpis = document.getElementById('kpis');
const compare = document.getElementById('compare');
const details = document.getElementById('details');

function fmt(n) {
  if (typeof n !== 'number' || Number.isNaN(n)) return '-';
  return n.toLocaleString();
}

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setKpiTone(el, value) {
  el.style.color = '#e9f1ff';
  if (typeof value !== 'number') return;
  if (value < 0) el.style.color = '#35d07f';
  if (value > 0) el.style.color = '#ff6b81';
}

function fmtSigned(n) {
  if (typeof n !== 'number' || Number.isNaN(n)) return '-';
  if (n === 0) return '0';
  return `${n > 0 ? '+' : '-'}${Math.abs(n).toLocaleString()}`;
}

function pctDelta(observerVal, baselineVal) {
  const base = typeof baselineVal === 'number' ? baselineVal : 0;
  const obs = typeof observerVal === 'number' ? observerVal : 0;
  if (base === 0) return obs === 0 ? 0 : 100;
  return ((obs - base) / Math.abs(base)) * 100;
}

function createBarRow(label, value, max, kind) {
  const row = document.createElement('div');
  row.className = 'bar-row';

  const info = document.createElement('div');
  info.className = 'bar-label';
  info.innerHTML = `<span>${label}</span><span>${fmt(value)}</span>`;

  const track = document.createElement('div');
  track.className = 'track';

  const fill = document.createElement('div');
  fill.className = `fill ${kind}`;
  const width = max > 0 ? Math.max(2, (value / max) * 100) : 0;
  fill.style.width = `${width}%`;

  track.appendChild(fill);
  row.appendChild(info);
  row.appendChild(track);
  return row;
}

function renderBars(containerId, observerVal, baselineVal) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';

  const obs = observerVal || 0;
  const base = baselineVal || 0;
  const delta = obs - base;
  const deltaPct = pctDelta(obs, base);

  const summary = document.createElement('div');
  summary.className = 'metric-summary';
  const deltaChip = document.createElement('span');
  deltaChip.className = 'delta-chip';
  if (delta > 0) deltaChip.classList.add('worse');
  if (delta < 0) deltaChip.classList.add('better');
  deltaChip.textContent = `${fmtSigned(delta)} (${deltaPct.toFixed(1)}%)`;
  summary.innerHTML = '<span class="muted">Observer - Baseline</span>';
  summary.appendChild(deltaChip);
  container.appendChild(summary);

  const max = Math.max(observerVal || 0, baselineVal || 0, 1);
  container.appendChild(createBarRow('Observer', obs, max, 'observer'));
  container.appendChild(createBarRow('Baseline', base, max, 'baseline'));

  // Difference lane: centered around zero so tiny gaps are still visible.
  const diffRow = document.createElement('div');
  diffRow.className = 'diff-row';
  const diffLabel = document.createElement('div');
  diffLabel.className = 'bar-label';
  diffLabel.innerHTML = '<span>Difference Lane</span><span>0 centered</span>';

  const track = document.createElement('div');
  track.className = 'diff-track';
  const center = document.createElement('div');
  center.className = 'diff-center';
  const fill = document.createElement('div');
  fill.className = `diff-fill ${delta >= 0 ? 'worse' : 'better'}`;

  const relative = Math.min(40, Math.max(2, Math.abs(deltaPct) * 0.7));
  fill.style.width = `${relative}%`;
  fill.style.left = delta >= 0 ? '50%' : `${50 - relative}%`;

  track.appendChild(center);
  track.appendChild(fill);
  diffRow.appendChild(diffLabel);
  diffRow.appendChild(track);
  container.appendChild(diffRow);
}

function renderNotes(id, notes) {
  const list = document.getElementById(id);
  list.innerHTML = '';
  if (!Array.isArray(notes) || notes.length === 0) {
    const li = document.createElement('li');
    li.textContent = 'No notes';
    list.appendChild(li);
    return;
  }
  notes.forEach((n) => {
    const li = document.createElement('li');
    li.textContent = String(n);
    list.appendChild(li);
  });
}

function render(report) {
  const observer = report.observer || {};
  const baseline = report.baseline || {};
  const delta = report.delta || {};

  metaSection.classList.remove('hidden');
  kpis.classList.remove('hidden');
  compare.classList.remove('hidden');
  details.classList.remove('hidden');

  setText('meta-project', observer.project || baseline.project || '-');
  setText('meta-task', report.task || observer.task || baseline.task || '-');
  setText('meta-model', report.model || observer.model || baseline.model || '-');
  setText('meta-reasoning', report.reasoning_effort || 'n/a');

  setText('kpi-events', fmt(delta.events));
  setText('kpi-tokens', fmt(delta.total_tokens));
  setText('kpi-rollbacks', fmt(delta.rollbacks));
  setText(
    'kpi-success',
    `${observer.success ? 'Observer OK' : 'Observer FAIL'} | ${baseline.success ? 'Baseline OK' : 'Baseline FAIL'}`
  );

  setKpiTone(document.getElementById('kpi-events'), delta.events);
  setKpiTone(document.getElementById('kpi-tokens'), delta.total_tokens);

  renderBars('bars-events', observer.total_events, baseline.total_events);
  renderBars('bars-tokens', observer.total_tokens, baseline.total_tokens);
  renderBars('bars-input', observer.total_input_tokens, baseline.total_input_tokens);
  renderBars('bars-output', observer.total_output_tokens, baseline.total_output_tokens);

  renderNotes('observer-notes', observer.notes);
  renderNotes('baseline-notes', baseline.notes);
}

function parseFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const report = JSON.parse(reader.result);
      if (!report || typeof report !== 'object' || !report.observer || !report.baseline) {
        throw new Error('Not a valid benchmark_report.json structure.');
      }
      dropHint.textContent = `Loaded: ${file.name}`;
      render(report);
    } catch (err) {
      dropHint.textContent = `Error: ${err.message}`;
    }
  };
  reader.readAsText(file);
}

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file) parseFile(file);
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) parseFile(file);
});
