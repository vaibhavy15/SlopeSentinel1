/* ============================================================
   SlopeSentinel — app.js  (frontend)
   ============================================================ */

'use strict';

/* ============================================================
   DASHBOARD — page switcher
   ============================================================ */
function switchPage(name, navEl) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + name).classList.add('active');
  if (navEl) navEl.classList.add('active');
}

/* ============================================================
   CHARTS (Chart.js)
   ============================================================ */
function initCharts() {
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];

  new Chart(document.getElementById('lineChart'), {
    type: 'line',
    data: {
      labels: months,
      datasets: [
        { label: 'Safe',     data: [42,43,47,44,48,45], borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,0.08)',   tension: 0.4, fill: true, pointRadius: 3 },
        { label: 'Caution',  data: [12,13,12,14,11,13], borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,0.06)',  tension: 0.4, fill: true, pointRadius: 3 },
        { label: 'Critical', data: [4, 3, 5, 3, 4, 3],  borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.05)',   tension: 0.4, fill: true, pointRadius: 3 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#7a8099', font: { size: 11 } } } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#7a8099', font: { size: 11 } } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#7a8099', font: { size: 11 } } }
      }
    }
  });

  new Chart(document.getElementById('donutChart'), {
    type: 'doughnut',
    data: {
      labels: ['Safe', 'Caution', 'Critical'],
      datasets: [{ data: [75, 18, 7], backgroundColor: ['#22c55e', '#f97316', '#ef4444'], borderColor: '#13161d', borderWidth: 3, hoverOffset: 6 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '72%',
      plugins: { legend: { position: 'bottom', labels: { color: '#7a8099', font: { size: 11 }, padding: 12 } } }
    }
  });
}

/* ============================================================
   MINE MAP — site marker selection
   ============================================================ */
function selectSite(name, risk, lat, lng, safe, riskIdx, stab) {
  document.getElementById('no-site-msg').style.display = 'none';
  const panel = document.getElementById('site-detail-panel');
  panel.classList.add('show');
  document.getElementById('sd-name').textContent   = name;
  const badge = document.getElementById('sd-badge');
  badge.textContent = risk.charAt(0).toUpperCase() + risk.slice(1);
  badge.className   = 'risk-badge ' + risk;
  document.getElementById('sd-coords').textContent  = lat + '°N, ' + lng + '°W';
  document.getElementById('sd-safe').textContent    = safe + '%';
  document.getElementById('sd-risk').textContent    = riskIdx;
  document.getElementById('sd-stab').textContent    = stab;
  document.getElementById('sd-coords2').textContent = 'Lat: ' + lat + ' | Lng: ' + lng;
}

/* ============================================================
   UPLOAD DATA
   ============================================================ */
const MOCK_RESULTS = [
  { id: 'Site-001', angle: 34, rain: 45,  density: 2.7, score: 0.12, pred: 'safe'     },
  { id: 'Site-002', angle: 52, rain: 120, density: 2.3, score: 0.61, pred: 'caution'  },
  { id: 'Site-003', angle: 68, rain: 210, density: 1.9, score: 0.87, pred: 'critical' },
  { id: 'Site-004', angle: 28, rain: 30,  density: 3.0, score: 0.08, pred: 'safe'     },
  { id: 'Site-005', angle: 47, rain: 95,  density: 2.5, score: 0.55, pred: 'caution'  }
];
const STAGES = ['Uploading…', 'Parsing CSV…', 'Running XGBoost model…', 'Generating predictions…', 'Done!'];

function dragOver(e)  { e.preventDefault(); document.getElementById('uploadZone').classList.add('drag'); }
function dragLeave()  { document.getElementById('uploadZone').classList.remove('drag'); }
function dropFile(e)  { e.preventDefault(); dragLeave(); if (e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]); }
function handleFile(i){ if (i.files[0]) processFile(i.files[0]); }

async function processFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  document.getElementById("progressArea").classList.add("show");

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    if (data.error) {
      alert(data.error);
      return;
    }

    console.log(data.results);
    alert("Prediction complete! Check console.");

  } catch (err) {
    console.error(err);
    alert("Upload failed");
  }
}

function showResults() {
  document.getElementById('uploadStatus').textContent = 'Analysis complete — ' + MOCK_RESULTS.length + ' predictions generated.';
  document.getElementById('resultsBody').innerHTML = MOCK_RESULTS.map(r => `
    <tr>
      <td style="font-family:monospace;font-size:12px">${r.id}</td>
      <td>${r.angle}°</td><td>${r.rain}</td><td>${r.density}</td>
      <td>${Math.round(r.score * 100)}%</td>
      <td><span class="risk-pill ${r.pred}">${r.pred.charAt(0).toUpperCase() + r.pred.slice(1)}</span></td>
    </tr>`).join('');
  document.getElementById('resultsTable').classList.add('show');
}

/* ============================================================
   ALERTS
   ============================================================ */
function viewAlertDetails() {
  switchPage('map', document.querySelector('.nav-item:nth-child(2)'));
}
