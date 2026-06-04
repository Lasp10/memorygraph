'use strict';

// ── Navigation ─────────────────────────────────────────────────────────────
const views = document.querySelectorAll('.view');
const navBtns = document.querySelectorAll('nav button[data-view]');

navBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.view;
    views.forEach(v => v.classList.toggle('active', v.id === target));
    navBtns.forEach(b => b.classList.toggle('active', b === btn));
    if (target === 'view-people') loadPeople();
  });
});

// ── Toast ──────────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, duration = 3000) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), duration);
}

// ── Stats ──────────────────────────────────────────────────────────────────
async function refreshStats() {
  try {
    const r = await fetch('/api/stats');
    const s = await r.json();
    document.getElementById('stat-photos').textContent = s.total_media ?? 0;
    document.getElementById('stat-processed').textContent = s.processed ?? 0;
    document.getElementById('stat-people').textContent = s.people ?? 0;
    document.getElementById('stat-events').textContent = s.events ?? 0;
  } catch (e) { /* ignore */ }
}

// ── Ingest View ────────────────────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const folderInput = document.getElementById('folder-input');
const processBtn = document.getElementById('process-btn');
const progressSection = document.getElementById('progress-section');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', async e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const files = [...e.dataTransfer.files];
  const zip = files.find(f => f.name.endsWith('.zip'));
  if (zip) {
    await uploadZip(zip);
  } else {
    showToast('Please drop a .zip file or use the folder path below.');
  }
});

dropZone.addEventListener('click', () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.zip';
  input.onchange = async () => {
    if (input.files[0]) await uploadZip(input.files[0]);
  };
  input.click();
});

async function uploadZip(file) {
  showToast(`Uploading ${file.name}…`);
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/api/ingest/zip', { method: 'POST', body: fd });
    const data = await r.json();
    showToast(`Ingested ${data.ingested} photos, skipped ${data.skipped}`);
    refreshStats();
  } catch (e) {
    showToast('Upload failed: ' + e.message);
  }
}

processBtn.addEventListener('click', async () => {
  const folder = folderInput.value.trim();
  if (!folder && !processBtn.dataset.ingest) {
    showToast('Enter a folder path first.');
    return;
  }

  processBtn.disabled = true;
  progressSection.classList.add('visible');

  const steps = ['step-ingest', 'step-faces', 'step-scenes', 'step-graph'];

  const setStep = (id, state) => {
    document.getElementById(id)?.classList.remove('active', 'done');
    if (state) document.getElementById(id)?.classList.add(state);
  };

  const setBar = pct => {
    document.getElementById('progress-bar').style.width = pct + '%';
  };

  try {
    // Step 1: Ingest
    if (folder) {
      setStep('step-ingest', 'active');
      const r1 = await fetch('/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: folder }),
      });
      const d1 = await r1.json();
      if (!r1.ok) throw new Error(d1.detail || 'Ingest failed');
      showToast(`Ingested ${d1.ingested} photos`);
    }
    setStep('step-ingest', 'done');
    setBar(25);

    // Step 2: Faces
    setStep('step-faces', 'active');
    const r2 = await fetch('/api/process/faces', { method: 'POST' });
    const d2 = await r2.json();
    if (!r2.ok) throw new Error(d2.detail || 'Face processing failed');
    setStep('step-faces', 'done');
    setBar(50);

    // Step 3: Scenes
    setStep('step-scenes', 'active');
    const r3 = await fetch('/api/process/scenes', { method: 'POST' });
    const d3 = await r3.json();
    if (!r3.ok) throw new Error(d3.detail || 'Scene tagging failed');
    setStep('step-scenes', 'done');
    setBar(75);

    // Step 4: Graph
    setStep('step-graph', 'active');
    await fetch('/api/graph');
    setStep('step-graph', 'done');
    setBar(100);

    await refreshStats();
    showToast('Processing complete!');

  } catch (e) {
    showToast('Error: ' + e.message, 5000);
    console.error(e);
  } finally {
    processBtn.disabled = false;
  }
});

// ── People View ────────────────────────────────────────────────────────────
async function loadPeople() {
  const grid = document.getElementById('people-grid');
  grid.innerHTML = '<p class="hint">Loading…</p>';
  try {
    const r = await fetch('/api/people');
    const people = await r.json();
    if (!people.length) {
      grid.innerHTML = '<div class="empty-state"><div class="icon">👥</div><p>No people found yet.<br>Run the pipeline from the Ingest tab first.</p></div>';
      return;
    }
    grid.innerHTML = '';
    people.forEach(person => grid.appendChild(buildPersonCard(person)));
  } catch (e) {
    grid.innerHTML = '<p style="color:var(--danger)">Failed to load people.</p>';
  }
}

function buildPersonCard(person) {
  const card = document.createElement('div');
  card.className = 'person-card';
  card.dataset.id = person.id;

  // Representative face
  const repFace = person.faces?.[0];
  let faceHtml = '';
  if (repFace?.thumbnail_path) {
    faceHtml = `<img class="face-thumb" src="/${repFace.thumbnail_path.replace(/^\//, '')}" onerror="this.style.display='none'">`;
  } else {
    faceHtml = `<div class="face-thumb" style="background:var(--surface2);border-radius:50%;"></div>`;
  }

  const isConfirmed = !!person.confirmed_name;

  card.innerHTML = `
    ${faceHtml}
    ${isConfirmed ? `<div class="confirmed-badge">✓ Confirmed</div>` : ''}
    <div class="person-name">${escHtml(person.name)}</div>
    <div class="person-count">${person.photo_count} photo${person.photo_count !== 1 ? 's' : ''}</div>
    <div class="name-input-row">
      <input type="text" placeholder="Enter real name…" value="${escHtml(person.confirmed_name || '')}">
      <button class="btn small success confirm-btn">Save</button>
    </div>
    <div class="face-gallery">
      ${(person.faces || []).map(f =>
        f.thumbnail_path
          ? `<img src="/${f.thumbnail_path.replace(/^\//, '')}" title="face" onerror="this.style.display='none'">`
          : ''
      ).join('')}
    </div>
    <button class="btn secondary small" style="width:100%;margin-top:8px;" data-toggle-gallery>
      Show faces
    </button>
  `;

  card.querySelector('.confirm-btn').addEventListener('click', async e => {
    e.stopPropagation();
    const input = card.querySelector('input');
    const name = input.value.trim();
    if (!name) { showToast('Enter a name first'); return; }
    try {
      const r = await fetch(`/api/people/${person.id}/name`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      if (!r.ok) throw new Error('Failed');
      showToast(`Saved: ${name}`);
      loadPeople();
    } catch (e) {
      showToast('Save failed: ' + e.message);
    }
  });

  card.querySelector('[data-toggle-gallery]').addEventListener('click', e => {
    e.stopPropagation();
    const gallery = card.querySelector('.face-gallery');
    const open = gallery.classList.toggle('open');
    e.target.textContent = open ? 'Hide faces' : 'Show faces';
  });

  return card;
}

// ── Search View ────────────────────────────────────────────────────────────
const searchInput = document.getElementById('search-input');
const searchBtn = document.getElementById('search-btn');
const resultsGrid = document.getElementById('results-grid');
let searchDebounce;

searchInput.addEventListener('input', () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(doSearch, 600);
});

searchBtn.addEventListener('click', doSearch);

searchInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') doSearch();
});

async function doSearch() {
  const query = searchInput.value.trim();
  if (!query) {
    resultsGrid.innerHTML = '';
    return;
  }
  resultsGrid.innerHTML = '<p class="hint">Searching…</p>';
  try {
    const r = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    const results = await r.json();
    renderResults(results);
  } catch (e) {
    resultsGrid.innerHTML = `<p style="color:var(--danger)">Search failed: ${e.message}</p>`;
  }
}

function renderResults(results) {
  if (!results.length) {
    resultsGrid.innerHTML = '<div class="empty-state"><div class="icon">🔍</div><p>No results found.</p></div>';
    return;
  }
  resultsGrid.innerHTML = '';
  results.forEach(item => {
    const card = document.createElement('div');
    card.className = 'result-card';
    const thumbSrc = item.thumbnail_path
      ? '/' + item.thumbnail_path.replace(/^\//, '')
      : '';
    const tagsHtml = (item.tags || []).map(t => `<span class="tag-chip">${escHtml(t)}</span>`).join('');
    const peopleStr = (item.people || []).join(', ');

    card.innerHTML = `
      <img class="thumb" src="${thumbSrc}" alt="" loading="lazy" onerror="this.style.display='none'">
      <div class="result-meta">
        ${item.date_taken ? `<div class="result-date">${item.date_taken.slice(0, 10)}</div>` : ''}
        ${peopleStr ? `<div class="result-people">${escHtml(peopleStr)}</div>` : ''}
        <div class="result-tags">${tagsHtml}</div>
      </div>
    `;

    card.addEventListener('click', () => openLightbox(item));
    resultsGrid.appendChild(card);
  });
}

// ── Lightbox ───────────────────────────────────────────────────────────────
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightbox-img');
const lightboxMeta = document.getElementById('lightbox-meta');

document.getElementById('lightbox-close').addEventListener('click', closeLightbox);
lightbox.addEventListener('click', e => { if (e.target === lightbox) closeLightbox(); });

function openLightbox(item) {
  lightboxImg.src = item.thumbnail_path ? '/' + item.thumbnail_path.replace(/^\//, '') : '';
  lightboxMeta.innerHTML = `
    <h3 style="margin-bottom:10px">${escHtml(item.people?.join(', ') || 'Unknown')}</h3>
    ${item.date_taken ? `<p class="hint">${item.date_taken.slice(0, 10)}</p>` : ''}
    <div class="result-tags" style="margin-top:10px">
      ${(item.tags || []).map(t => `<span class="tag-chip">${escHtml(t)}</span>`).join('')}
    </div>
    <p style="margin-top:12px;font-size:12px;color:var(--text-muted);word-break:break-all">${escHtml(item.filepath || '')}</p>
  `;
  lightbox.classList.add('open');
}

function closeLightbox() {
  lightbox.classList.remove('open');
  lightboxImg.src = '';
}

// ── Utils ──────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Init ───────────────────────────────────────────────────────────────────
refreshStats();
setInterval(refreshStats, 10000);
