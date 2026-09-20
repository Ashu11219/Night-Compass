const LOW_CONFIDENCE = 0.3;  // same threshold as run_demo.py
const root = document.documentElement;

/* ---------- Theme switch ---------- */
const themeToggle = document.getElementById('theme-toggle');

function applyTheme(theme) {
  root.dataset.theme = theme;
  themeToggle.setAttribute('aria-checked', String(theme === 'dark'));
}

let savedTheme = null;
try { savedTheme = localStorage.getItem('nc-theme'); } catch (e) { /* storage blocked */ }
applyTheme(savedTheme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));

themeToggle.addEventListener('click', () => {
  const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  try { localStorage.setItem('nc-theme', next); } catch (e) { /* storage blocked */ }
});

/* ---------- Tabs ---------- */
const tabs = document.querySelectorAll('[role="tab"]');

tabs.forEach(tab => tab.addEventListener('click', () => {
  tabs.forEach(t => {
    const selected = t === tab;
    t.setAttribute('aria-selected', String(selected));
    document.getElementById(t.getAttribute('aria-controls')).hidden = !selected;
  });
  if (tab.id !== 'tab-video') stopStream();  // don't keep streaming in a hidden tab
}));

/* ---------- Shared helpers ---------- */
function setupDropzone(zone, input, onFile) {
  input.addEventListener('change', () => {
    if (input.files[0]) onFile(input.files[0]);
  });
  ['dragenter', 'dragover'].forEach(evt => zone.addEventListener(evt, e => {
    e.preventDefault();
    zone.classList.add('is-over');
  }));
  ['dragleave', 'drop'].forEach(evt => zone.addEventListener(evt, e => {
    e.preventDefault();
    zone.classList.remove('is-over');
  }));
  zone.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  });
}

function setStatus(el, text, isError = false) {
  el.textContent = text;
  el.classList.toggle('is-error', isError);
}

function showInFrame(img, emptyText, src) {
  img.src = src;
  img.hidden = false;
  emptyText.hidden = true;
}

function clearFrame(img, emptyText) {
  img.removeAttribute('src');
  img.hidden = true;
  emptyText.hidden = false;
}

/* ---------- Image detection ---------- */
const imageZone = document.getElementById('image-zone');
const imageInput = document.getElementById('image-input');
const imageFileName = document.getElementById('image-file-name');
const imageRun = document.getElementById('image-run');
const imageStatus = document.getElementById('image-status');
const imageResults = document.getElementById('image-results');
const imageView = document.getElementById('image-view');
const imageEmpty = document.getElementById('image-empty');

let imageFile = null;

setupDropzone(imageZone, imageInput, file => {
  if (!file.type.startsWith('image/')) {
    setStatus(imageStatus, 'That file is not an image. Choose a JPG or PNG.', true);
    return;
  }
  imageFile = file;
  imageFileName.textContent = file.name;
  imageResults.replaceChildren();
  showInFrame(imageView, imageEmpty, URL.createObjectURL(file));
  imageRun.disabled = false;
  setStatus(imageStatus, 'Showing the original. Run detection to get the annotated version.');
});

imageRun.addEventListener('click', async () => {
  imageRun.disabled = true;
  setStatus(imageStatus, 'Detecting objects…');

  const form = new FormData();
  form.append('file', imageFile);

  try {
    const res = await fetch('/detect/image', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server responded with ${res.status}.`);

    showInFrame(imageView, imageEmpty, `data:image/jpeg;base64,${data.image}`);
    renderImageResults(data);

    const count = data.detections.length;
    setStatus(imageStatus, count ? `Found ${count} object${count === 1 ? '' : 's'}.` : 'No objects found in this image.');
  } catch (err) {
    setStatus(imageStatus, `Detection failed. ${err.message} Check that the server is running.`, true);
  } finally {
    imageRun.disabled = false;
  }
});

function renderImageResults(data) {
  imageResults.replaceChildren();
  if (!data.detections.length) return;

  const source = document.createElement('span');
  source.className = 'source';
  source.textContent = `Source: ${data.source === 'THERMAL' ? 'Thermal model' : 'RGB model'}`;
  imageResults.append(source);

  data.alerts.forEach(alert => {
    const p = document.createElement('p');
    p.className = 'alert';
    p.textContent = `Close ${alert.class}, ${alert.direction}`;
    imageResults.append(p);
  });

  const list = document.createElement('ul');
  list.className = 'detections';

  [...data.detections]
    .sort((a, b) => b.confidence - a.confidence)
    .forEach(det => {
      const li = document.createElement('li');
      const name = document.createElement('span');
      const conf = document.createElement('span');
      conf.className = 'conf';
      conf.textContent = `${Math.round(det.confidence * 100)}%`;

      name.className = det.confidence >= LOW_CONFIDENCE ? 'name' : 'name is-low';
      name.textContent = det.class;

      li.append(name, conf);
      list.append(li);
    });

  imageResults.append(list);
}

/* ---------- Video detection ---------- */
const videoZone = document.getElementById('video-zone');
const videoInput = document.getElementById('video-input');
const videoFileName = document.getElementById('video-file-name');
const videoRun = document.getElementById('video-run');
const videoStop = document.getElementById('video-stop');
const videoStatus = document.getElementById('video-status');
const videoView = document.getElementById('video-view');
const videoEmpty = document.getElementById('video-empty');
const videoLive = document.getElementById('video-live');

let videoFile = null;

setupDropzone(videoZone, videoInput, file => {
  if (!file.type.startsWith('video/')) {
    setStatus(videoStatus, 'That file is not a video. Choose an MP4, AVI or MOV.', true);
    return;
  }
  stopStream();
  videoFile = file;
  videoFileName.textContent = file.name;
  videoRun.disabled = false;
  setStatus(videoStatus, 'Ready. Start detection to play the annotated feed.');
});

videoRun.addEventListener('click', async () => {
  stopStream();
  videoRun.disabled = true;
  setStatus(videoStatus, 'Uploading video…');

  const form = new FormData();
  form.append('file', videoFile);

  try {
    const res = await fetch('/detect/video', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `Server responded with ${res.status}.`);

    showInFrame(videoView, videoEmpty, `/stream/${encodeURIComponent(data.video_id)}`);
    videoLive.hidden = false;
    videoStop.hidden = false;
    setStatus(videoStatus, 'Detection running. The feed freezes on the last frame when the video ends.');
  } catch (err) {
    setStatus(videoStatus, `Upload failed. ${err.message} Check that the server is running.`, true);
  } finally {
    videoRun.disabled = false;
  }
});

videoStop.addEventListener('click', () => {
  stopStream();
  setStatus(videoStatus, 'Feed stopped. Start detection again to replay it.');
});

function stopStream() {
  clearFrame(videoView, videoEmpty);  // dropping the src closes the stream connection
  videoLive.hidden = true;
  videoStop.hidden = true;
}
document.getElementById('tab-video').click();