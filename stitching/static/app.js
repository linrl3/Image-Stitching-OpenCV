const messages = {
  zh: {
    title: '图片拼接', language: '语言', github: '查看 GitHub 仓库（在新标签页打开）', left: '左图', right: '右图',
    choose: '选择图片', chooseLeft: '选择左图', chooseRight: '选择右图', swap: '交换左右',
    leftPreview: '左图预览', rightPreview: '右图预览', panoramaPreview: '全景预览', matchingPreview: '特征匹配图',
    limits: '左图 ≥ 800 px · 合计 ≤ 32 MiB', sample: '试用示例', stitch: '拼接',
    download: '下载 PNG', report: '下载诊断报告', details: '详情',
    matches: '匹配点', inliers: '内点比例', error: '中位误差',
    clipped: '部分边缘超出画布，可能被裁剪。', processing: '正在拼接…',
    connection: '连接失败，请确认服务正在运行后重试。',
    invalid_request: '请选择左右两张图片。', invalid_image: '请使用有效的图片文件。',
    decode_failed: '无法读取图片，请换一个文件。', image_too_small: '左图宽度至少需要 800 像素。',
    insufficient_features: '图片细节不足，请换一组纹理更丰富的照片。',
    insufficient_matches: '有效匹配不足，请选择重叠更多的照片。',
    invalid_homography: '无法对齐图片，请选择重叠更多的照片。',
    empty_crop: '无法生成有效结果，请换一组照片。',
    image_too_large: '每张图片最多支持 1200 万像素。', upload_too_large: '两张图片合计不能超过 32 MiB。',
    busy: '已有任务正在运行，请稍后重试。', invalid_origin: '请从当前页面重新提交图片。',
    sample_unavailable: '示例图片不可用，请自行选择图片。',
    processing_error: '拼接失败，请换一组图片或查看诊断报告。',
  },
  en: {
    title: 'Image Stitching', language: 'Language', github: 'View repository on GitHub (opens in a new tab)', left: 'Left', right: 'Right',
    choose: 'Choose image', chooseLeft: 'Choose left image', chooseRight: 'Choose right image', swap: 'Swap images',
    leftPreview: 'Left image preview', rightPreview: 'Right image preview', panoramaPreview: 'Panorama preview', matchingPreview: 'Feature matches',
    limits: 'Left ≥ 800 px · Total ≤ 32 MiB', sample: 'Try example', stitch: 'Stitch',
    download: 'Download PNG', report: 'Download report', details: 'Details',
    matches: 'Matches', inliers: 'Inlier ratio', error: 'Median error',
    clipped: 'Some edges extend beyond the canvas and may be cropped.', processing: 'Stitching…',
    connection: 'Connection failed. Check that the server is running and retry.',
    invalid_request: 'Choose both a left and a right image.', invalid_image: 'Please use valid image files.',
    decode_failed: 'Cannot read this image. Choose another file.', image_too_small: 'The left image must be at least 800 pixels wide.',
    insufficient_features: 'Not enough detail. Try photos with more texture.',
    insufficient_matches: 'Not enough matches. Try photos with more overlap.',
    invalid_homography: 'Cannot align these images. Try photos with more overlap.',
    empty_crop: 'Cannot produce a valid result. Try another pair of photos.',
    image_too_large: 'Each image must be 12 megapixels or smaller.', upload_too_large: 'The two images must total no more than 32 MiB.',
    busy: 'Another stitch is running. Please retry shortly.', invalid_origin: 'Please submit the images again from this page.',
    sample_unavailable: 'The example is unavailable. Choose your own images.',
    processing_error: 'Stitching failed. Try another pair or check the report.',
  },
};

const $ = (id) => document.getElementById(id);
const selected = [null, null];
const previewUrls = [null, null];
let language = navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en';
try {
  const saved = localStorage.getItem('stitching.language');
  if (Object.hasOwn(messages, saved)) language = saved;
} catch { /* Language switching also works when browser storage is unavailable. */ }
let busy = false;
let statusKey = '';
let report = null;
let reportUrl = null;
let sampleInputs = null;
let sampleReversed = false;

const t = (key) => messages[language][key] || messages[language].processing_error;
const ready = () => sampleInputs !== null || selected.every(Boolean);

function setStatus(key = '', state = '') {
  statusKey = key;
  $('status').textContent = key ? t(key) : '';
  $('status').className = state;
}

function renderMetrics() {
  if (!report || report.status !== 'success') return;
  const number = (value, digits) => value.toLocaleString(language, {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  });
  $('output-size').textContent = `${report.output.width} × ${report.output.height} · ${number(report.timings_seconds.compute_total, 2)} s`;
  $('metric-matches').textContent = number(report.matches.ratio_passed, 0);
  $('metric-ratio').textContent = `${number(report.registration.inlier_ratio * 100, 1)}%`;
  const error = report.registration.median_inlier_error_px;
  $('metric-error').textContent = error == null ? '—' : `${number(error, 2)} px`;
}

function setLanguage(next) {
  language = next;
  document.documentElement.lang = next === 'zh' ? 'zh-CN' : 'en';
  document.title = t('title');
  document.querySelectorAll('[data-i18n]').forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelectorAll('[data-i18n-aria]').forEach((element) => {
    element.setAttribute('aria-label', t(element.dataset.i18nAria));
  });
  document.querySelectorAll('[data-i18n-alt]').forEach((element) => {
    element.alt = t(element.dataset.i18nAlt);
  });
  ['zh', 'en'].forEach((lang) => $(`lang-${lang}`).setAttribute('aria-pressed', String(lang === next)));
  $('status').textContent = statusKey ? t(statusKey) : '';
  renderMetrics();
  try { localStorage.setItem('stitching.language', next); } catch { /* Optional persistence. */ }
}

function controls() {
  $('submit').disabled = busy || !ready();
  $('swap').disabled = busy || !ready();
  ['image1', 'image2', 'sample'].forEach((id) => { if ($(id)) $(id).disabled = busy; });
  $('stitch-form').setAttribute('aria-busy', String(busy));
}

function clearResults() {
  $('results').hidden = true;
  $('details').open = false;
  $('error-report').hidden = true;
  report = null;
  if (reportUrl) URL.revokeObjectURL(reportUrl);
  reportUrl = null;
  setStatus();
}

function showInput(index) {
  const file = selected[index];
  const sample = sampleInputs?.[index];
  if (previewUrls[index]) URL.revokeObjectURL(previewUrls[index]);
  previewUrls[index] = file ? URL.createObjectURL(file) : null;
  const source = sample?.url || previewUrls[index];
  const preview = $(`preview${index + 1}`);
  preview.hidden = !source;
  if (source) preview.src = source; else preview.removeAttribute('src');
  $(index === 0 ? 'left-card' : 'right-card').classList.toggle('selected', !!source);
  $(`filename${index + 1}`).textContent = sample?.name || file?.name || '';
}

['image1', 'image2'].forEach((id, index) => $(id).addEventListener('change', (event) => {
  if (sampleInputs) {
    sampleInputs = null;
    sampleReversed = false;
    showInput(0); showInput(1);
  }
  selected[index] = event.target.files[0] || null;
  clearResults(); showInput(index); controls();
}));

$('swap').addEventListener('click', () => {
  if (sampleInputs) {
    sampleReversed = !sampleReversed;
    sampleInputs.reverse();
  } else {
    selected.reverse();
    $('image1').value = ''; $('image2').value = '';
  }
  showInput(0); showInput(1); clearResults(); controls();
});

function saveReport(next) {
  report = next;
  reportUrl = URL.createObjectURL(new Blob([JSON.stringify(next, null, 2)], {type: 'application/json'}));
  ['download-report', 'error-report'].forEach((id) => {
    $(id).href = reportUrl;
    $(id).download = 'report.json';
  });
}

async function run(sample = sampleInputs !== null) {
  if (busy) return;
  clearResults();
  if (!sample && !ready()) { setStatus('invalid_request', 'error'); return; }
  if (!sample && selected.reduce((sum, file) => sum + file.size, 0) > 32 * 1024 * 1024) {
    setStatus('upload_too_large', 'error'); return;
  }
  busy = true; controls(); setStatus('processing', 'busy');
  const data = new FormData();
  data.set('matching', 'true');
  if (sample) data.set('reverse', String(sampleReversed));
  else { data.set('image1', selected[0]); data.set('image2', selected[1]); }
  try {
    const response = await fetch(sample ? '/api/sample' : '/api/stitch', {method: 'POST', body: data});
    const payload = await response.json().catch(() => null);
    if (payload?.report) saveReport(payload.report);
    if (!response.ok || !payload?.files) {
      const code = payload?.report?.error?.code || (response.status === 413 ? 'upload_too_large' : 'processing_error');
      setStatus(code, 'error'); $('error-report').hidden = !reportUrl; return;
    }
    const files = payload.files;
    $('result-image').src = files['panorama.png'];
    $('download-panorama').href = `${files['panorama.png']}?download=1`;
    $('matching-image').hidden = !files['matching.jpg'];
    if (files['matching.jpg']) $('matching-image').src = files['matching.jpg'];
    $('geometry-note').hidden = !report.geometry?.extends_beyond_canvas;
    if (sample) {
      selected.fill(null); $('image1').value = ''; $('image2').value = '';
      sampleInputs = [1, 2].map((id) => ({url: files[`input${id}.jpg`], name: report.source.names[id - 1]}));
      showInput(0); showInput(1);
    }
    renderMetrics(); $('results').hidden = false; setStatus();
    $('results').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'nearest'});
  } catch {
    setStatus('connection', 'error');
    $('error-report').hidden = !reportUrl;
  } finally {
    busy = false; controls();
  }
}

$('stitch-form').addEventListener('submit', (event) => { event.preventDefault(); run(); });
if ($('sample')) $('sample').addEventListener('click', () => { sampleReversed = false; run(true); });
['zh', 'en'].forEach((lang) => $(`lang-${lang}`).addEventListener('click', () => setLanguage(lang)));
setLanguage(language);
controls();
