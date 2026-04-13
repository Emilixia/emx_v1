'use strict';

const dropZone = document.getElementById('drop-zone');
const dropPlaceholder = document.getElementById('drop-placeholder');
const previewImg = document.getElementById('preview-img');
const btnSelect = document.getElementById('btn-select');
const btnAnalyze = document.getElementById('btn-analyze');

const resultsPlaceholder = document.getElementById('results-placeholder');
const resultsContent = document.getElementById('results-content');
const loadingEl = document.getElementById('loading');
const loadingText = document.getElementById('loading-text');
const errorBox = document.getElementById('error-box');
const errorMessage = document.getElementById('error-message');

const flowerResults = document.getElementById('flower-results');
const stemCountValue = document.getElementById('stem-count-value');
const stemMethod = document.getElementById('stem-method');

const wikiLoading = document.getElementById('wiki-loading');
const wikiBody = document.getElementById('wiki-body');
const wikiNone = document.getElementById('wiki-none');
const wikiExtract = document.getElementById('wiki-extract');
const wikiThumbnail = document.getElementById('wiki-thumbnail');
const wikiReadMore = document.getElementById('wiki-read-more');

let currentImagePath = null;

// ─── Image loading helpers ──────────────────────────────────

function loadImage(filePath) {
  currentImagePath = filePath;
  previewImg.src = 'file://' + filePath;
  previewImg.classList.remove('hidden');
  dropPlaceholder.classList.add('hidden');
  btnAnalyze.disabled = false;
  resetResults();
}

function resetResults() {
  resultsContent.classList.add('hidden');
  errorBox.classList.add('hidden');
  loadingEl.classList.add('hidden');
  resultsPlaceholder.classList.remove('hidden');
  // Reset wiki card
  wikiLoading.classList.add('hidden');
  wikiBody.classList.add('hidden');
  wikiNone.classList.add('hidden');
  wikiReadMore.onclick = null;
}

// ─── Button: Select image ──────────────────────────────────

btnSelect.addEventListener('click', async () => {
  const filePath = await window.flowerAPI.selectImage();
  if (filePath) loadImage(filePath);
});

// ─── Drop zone: click to select ───────────────────────────

dropZone.addEventListener('click', async () => {
  const filePath = await window.flowerAPI.selectImage();
  if (filePath) loadImage(filePath);
});

dropZone.addEventListener('keydown', async (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    const filePath = await window.flowerAPI.selectImage();
    if (filePath) loadImage(filePath);
  }
});

// ─── Drop zone: drag-and-drop ──────────────────────────────

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && /\.(jpe?g|png|bmp|webp|tiff?)$/i.test(file.name)) {
    loadImage(file.path);
  }
});

// ─── Button: Analyze ──────────────────────────────────────

btnAnalyze.addEventListener('click', async () => {
  if (!currentImagePath) return;

  // Show loading
  resultsPlaceholder.classList.add('hidden');
  resultsContent.classList.add('hidden');
  errorBox.classList.add('hidden');
  loadingEl.classList.remove('hidden');
  btnAnalyze.disabled = true;

  try {
    // Phase 1: ensure Python dependencies are installed (fast on subsequent runs).
    loadingText.textContent = 'Checking Python dependencies…';
    const { alreadyInstalled } = await window.flowerAPI.ensureDeps();
    if (!alreadyInstalled) {
      // Deps were just installed — update text while we continue.
      loadingText.textContent = 'Dependencies installed. Analyzing image…';
    } else {
      loadingText.textContent = 'Analyzing image…';
    }

    // Phase 2: run the analysis.
    const result = await window.flowerAPI.analyzeImage(currentImagePath);
    renderResults(result);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    loadingEl.classList.add('hidden');
    loadingText.textContent = 'Analyzing image…';
    btnAnalyze.disabled = false;
  }
});

// ─── Render results ───────────────────────────────────────

function renderResults(data) {
  // Flower identification
  flowerResults.innerHTML = '';
  const predictions = data.flowers || [];
  predictions.forEach((pred, i) => {
    const pct = Math.round((pred.score || 0) * 100);
    const item = document.createElement('div');
    item.className = 'flower-item' + (i === 0 ? ' top' : '');
    item.innerHTML = `
      <span class="flower-item-rank">${i + 1}.</span>
      <span class="flower-item-label">${escapeHtml(pred.label)}</span>
      <div class="flower-item-bar-wrap">
        <div class="flower-item-bar" style="width:${pct}%"></div>
      </div>
      <span class="flower-item-pct">${pct}%</span>
    `;
    flowerResults.appendChild(item);
  });

  if (predictions.length === 0) {
    flowerResults.innerHTML = '<p style="color:var(--text-muted);font-size:0.85rem">No flower predictions returned.</p>';
  }

  // Stem count
  const stemCount = data.stem_count != null ? data.stem_count : '—';
  stemCountValue.textContent = stemCount;
  stemMethod.textContent = data.stem_method || '';

  resultsPlaceholder.classList.add('hidden');
  resultsContent.classList.remove('hidden');

  // Kick off Wikipedia lookup for the top prediction
  if (predictions.length > 0) {
    fetchFlowerInfo(predictions[0].label);
  }
}

function showError(msg) {
  errorMessage.textContent = msg;
  errorBox.classList.remove('hidden');
  resultsPlaceholder.classList.add('hidden');
}

// ─── Wikipedia lookup ─────────────────────────────────────

async function fetchFlowerInfo(rawLabel) {
  // Show loading state inside the wiki card
  wikiLoading.classList.remove('hidden');
  wikiBody.classList.add('hidden');
  wikiNone.classList.add('hidden');

  // Normalise the label: replace underscores/hyphens with spaces, title-case
  const searchTerm = rawLabel.replace(/[_-]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  try {
    const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(searchTerm)}`;
    const response = await fetch(url, { headers: { Accept: 'application/json' } });

    if (!response.ok) {
      // Try a broader search via the opensearch API to find the best match
      const searchUrl = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(searchTerm.split(' ')[0])}`;
      const fallback = await fetch(searchUrl, { headers: { Accept: 'application/json' } });
      if (!fallback.ok) throw new Error('Not found');
      renderWikiResult(await fallback.json());
    } else {
      renderWikiResult(await response.json());
    }
  } catch (_err) {
    wikiLoading.classList.add('hidden');
    wikiNone.classList.remove('hidden');
  }
}

function renderWikiResult(data) {
  wikiLoading.classList.add('hidden');

  const extract = data.extract || '';
  const pageUrl = data.content_urls && data.content_urls.desktop && data.content_urls.desktop.page;
  const thumb = data.thumbnail && data.thumbnail.source;

  if (!extract && !pageUrl) {
    wikiNone.classList.remove('hidden');
    return;
  }

  wikiExtract.textContent = extract || 'No description available.';

  if (thumb) {
    wikiThumbnail.src = thumb;
    wikiThumbnail.alt = data.title || 'Flower';
    wikiThumbnail.classList.remove('hidden');
  } else {
    wikiThumbnail.classList.add('hidden');
  }

  if (pageUrl) {
    wikiReadMore.classList.remove('hidden');
    wikiReadMore.onclick = () => window.flowerAPI.openUrl(pageUrl);
  } else {
    wikiReadMore.classList.add('hidden');
  }

  wikiBody.classList.remove('hidden');
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
