/* ============================================================
   SlopeSentinel — app.js (frontend utilities)
   ============================================================ */
'use strict';

/* drag-and-drop helpers (used by upload.html) */
function dragOver(e)  { e.preventDefault(); document.getElementById('uploadZone')?.classList.add('drag'); }
function dragLeave()  { document.getElementById('uploadZone')?.classList.remove('drag'); }
function dropFile(e)  {
  e.preventDefault(); dragLeave();
  const f = e.dataTransfer?.files[0];
  if (!f) return;
  const dt = new DataTransfer(); dt.items.add(f);
  const inp = document.getElementById('fileInput');
  if (inp) { inp.files = dt.files; showFileName(inp); }
}