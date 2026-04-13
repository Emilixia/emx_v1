'use strict';

const dropZone = document.getElementById('drop-zone');
const dropPlaceholder = document.getElementById('drop-placeholder');
const previewImg = document.getElementById('preview-img');
const btnSelect = document.getElementById('btn-select');
const btnAnalyze = document.getElementById('btn-analyze');

const resultsPlaceholder = document.getElementById('results-placeholder');
const resultsContent = document.getElementById('results-content');
const loadingEl = document.getElementById('loading');
const errorBox = document.getElementById('error-box');
const errorMessage = document.getElementById('error-message');

const flowerResults = document.getElementById('flower-results');
const stemCountValue = document.getElementById('stem-count-value');
const stemMethod = document.getElementById('stem-method');

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
    const result = await window.flowerAPI.analyzeImage(currentImagePath);
    renderResults(result);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    loadingEl.classList.add('hidden');
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
}

function showError(msg) {
  errorMessage.textContent = msg;
  errorBox.classList.remove('hidden');
  resultsPlaceholder.classList.add('hidden');
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
