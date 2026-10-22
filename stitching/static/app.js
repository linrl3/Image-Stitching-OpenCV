const $ = (id) => document.getElementById(id);
const form = $('stitch-form');
const selected = [null, null];
const previewUrls = [null, null];
let busy = false;
let resultFiles = {};
let reportUrl = null;
let sampleShown = false;
let sampleReversed = false;

function status(message, state = '') { $('status').textContent = message; $('status').className = state; }
function controls() {
  $('submit').disabled = busy || (!sampleShown && !selected.every(Boolean));
  ['image1', 'image2', 'matching', 'swap', 'sample'].forEach((id) => { if ($(id)) $(id).disabled = busy; });
}
function clearResults() {
  $('results').hidden = true; $('diagnostics').hidden = true; resultFiles = {};
  if (reportUrl) { URL.revokeObjectURL(reportUrl); reportUrl = null; }
}
function showInput(index) {
  const file = selected[index]; const id = index + 1;
  if (previewUrls[index]) URL.revokeObjectURL(previewUrls[index]);
  previewUrls[index] = file ? URL.createObjectURL(file) : null;
  const preview = $(`preview${id}`); preview.hidden = !file;
  if (file) preview.src = previewUrls[index]; else preview.removeAttribute('src');
  $(index === 0 ? 'left-card' : 'right-card').classList.toggle('selected', !!file);
  $(`filename${id}`).textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MiB` : '尚未选择';
}
['image1', 'image2'].forEach((id, index) => $(id).addEventListener('change', (event) => {
  if (sampleShown) { sampleShown = false; sampleReversed = false; showInput(0); showInput(1); }
  selected[index] = event.target.files[0] || null;
  clearResults(); showInput(index); controls(); status('照片已更新，点击开始拼接。');
}));
$('swap').addEventListener('click', () => {
  if (sampleShown) {
    sampleReversed = !sampleReversed;
    [$('preview1').src, $('preview2').src] = [$('preview2').src, $('preview1').src];
    [$('filename1').textContent, $('filename2').textContent] = [$('filename2').textContent, $('filename1').textContent];
    clearResults(); controls(); status('已交换示例照片，点击开始拼接。'); return;
  }
  sampleShown = false;
  [selected[0], selected[1]] = [selected[1], selected[0]];
  $('image1').value = ''; $('image2').value = '';
  showInput(0); showInput(1); clearResults(); controls(); status('已交换左右照片。');
});
function showReport(report) {
  $('diagnostics').hidden = false;
  $('report-text').textContent = JSON.stringify(report, null, 2);
  const labels = {features:'提取特征', matching:'匹配', homography:'估计变换', diagnostics:'诊断', blend_left:'左图融合', warp:'图像变换', blend_right:'右图融合', crop:'裁剪', matching_visualization:'匹配图', read:'读取', encode:'编码'};
  $('stage-times').replaceChildren();
  Object.entries(report.timings_seconds || {}).forEach(([key, value]) => {
    if (!labels[key]) return;
    const item = document.createElement('span'); item.textContent = `${labels[key]} ${(value * 1000).toFixed(0)} ms`; $('stage-times').append(item);
  });
  reportUrl = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type:'application/json'}));
  $('download-report').href = reportUrl; $('download-report').download = 'report.json';
  $('download-diagnostics').href = reportUrl; $('download-diagnostics').download = 'report.json';
}
function switchTab(matching) {
  const name = matching ? 'matching.jpg' : 'panorama.png';
  if (!resultFiles[name]) return;
  $('result-image').src = resultFiles[name]; $('result-image').alt = matching ? '特征匹配预览' : '拼接全景预览';
  ['panorama-tab','matching-tab'].forEach((id, index) => {
    const active = index === Number(matching); $(id).classList.toggle('active', active); $(id).setAttribute('aria-selected', String(active));
  });
  $('result-panel').setAttribute('aria-labelledby', matching ? 'matching-tab' : 'panorama-tab');
}
$('panorama-tab').addEventListener('click', () => switchTab(false));
$('matching-tab').addEventListener('click', () => switchTab(true));
async function run(sample = sampleShown) {
  if (busy) return;
  if (!sample && selected.reduce((sum, file) => sum + (file?.size || 0), 0) > 32 * 1024 * 1024) { status('两张图片合计超过 32 MiB，请选择较小的文件。', 'error'); return; }
  clearResults(); busy = true; controls(); status('正在提取特征、匹配并融合，请稍候…', 'busy');
  const data = new FormData(); data.set('matching', String($('matching').checked));
  if (sample) data.set('reverse', String(sampleReversed));
  if (!sample) { data.set('image1', selected[0]); data.set('image2', selected[1]); }
  try {
    const response = await fetch(sample ? '/api/sample' : '/api/stitch', {method:'POST', body:data});
    const payload = await response.json();
    if (payload.report) showReport(payload.report);
    if (!response.ok) {
      const messages = {image_too_small:'左图宽度至少需要 800 像素。', insufficient_features:'照片缺少可匹配的纹理，请换一组细节更丰富的图片。', insufficient_matches:'两图的有效匹配不足，请选择重叠更多的照片。', invalid_homography:'无法估计可靠的图像变换，请检查照片的重叠区域。', empty_crop:'当前裁剪方式无法找到有效区域，请换一组照片。', decode_failed:'无法读取图片，请选择有效的图像文件。', image_too_large:'每张图片最多支持 1200 万像素，请先缩小图片。', upload_too_large:'两张图片合计超过 32 MiB。', busy:'已有拼接任务运行，请稍后重试。'};
      status(`未能完成拼接：${messages[payload.report?.error?.code] || payload.error || '请检查图片并重试。'}`, 'error'); return;
    }
    resultFiles = payload.files; const report = payload.report;
    $('results').hidden = false; $('matching-tab').hidden = !resultFiles['matching.jpg'];
    $('download-panorama').href = `${resultFiles['panorama.png']}?download=1`;
    $('output-size').textContent = `${report.output.width} × ${report.output.height} px`;
    $('metric-matches').textContent = report.matches.ratio_passed;
    $('metric-ratio').textContent = `${(report.registration.inlier_ratio * 100).toFixed(1)}%`;
    $('metric-error').textContent = report.registration.median_inlier_error_px == null ? '—' : `${report.registration.median_inlier_error_px.toFixed(2)} px`;
    $('metric-time').textContent = `${report.timings_seconds.compute_total.toFixed(2)} s`;
    $('geometry-note').hidden = !report.geometry?.extends_beyond_canvas;
    if (sample) {
      selected.fill(null); $('image1').value = ''; $('image2').value = '';
      showInput(0); showInput(1); sampleShown = true;
      [1, 2].forEach((id) => { const preview = $(`preview${id}`); preview.src = resultFiles[`input${id}.jpg`]; preview.hidden = false; $(id === 1 ? 'left-card' : 'right-card').classList.add('selected'); $(`filename${id}`).textContent = report.source.names[id - 1]; });
    }
    switchTab(false); status('拼接完成，可以查看匹配图或下载全景和报告。');
    $('results').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block:'start'});
  } catch (error) { status('连接中断或服务不可用，请确认本地服务仍在运行后重试。', 'error'); }
  finally { busy = false; controls(); }
}
form.addEventListener('submit', (event) => { event.preventDefault(); run(); });
if ($('sample')) $('sample').addEventListener('click', () => { sampleReversed = false; run(true); });
