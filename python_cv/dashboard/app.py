import logging

from flask import Flask, jsonify, render_template_string
from database.database import Database
from config.settings import FLASK_HOST, FLASK_PORT, FLASK_DEBUG

app = Flask(__name__)
_db = None

def init_dashboard(db):
    global _db
    _db = db

@app.route("/api/entries")
def api_entries():
    return jsonify(_db.get_today_entries())

@app.route("/api/anomalies")
def api_anomalies():
    return jsonify(_db.get_recent_anomalies())

@app.route("/api/rfid")
def api_rfid():
    return jsonify(_db.get_rfid_events())

@app.route("/api/presence")
def api_presence():
    return jsonify(_db.get_current_presence())

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Warehouse Access Monitor</title>
<style>
:root{--bg:#f3f5f9;--card:#fff;--ink:#1f2937;--muted:#6b7280;--line:#e8ebf0;--blue:#2563eb;--indigo:#4f46e5;--green:#16a34a;--amber:#d97706;--red:#dc2626;--shadow:0 1px 3px rgba(16,24,40,.08),0 1px 2px rgba(16,24,40,.05)}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font-family:'Segoe UI',system-ui,Arial,sans-serif;padding-bottom:40px}
header{background:#fff;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10;box-shadow:var(--shadow)}
header .in{max-width:1200px;margin:0 auto;padding:14px 24px;display:flex;align-items:center;justify-content:space-between}
header .brand{display:flex;align-items:center;gap:12px}
header .logo{width:42px;height:42px;border-radius:11px;background:linear-gradient(135deg,#2563eb,#4f46e5);display:grid;place-items:center;font-size:21px}
header b{font-size:1.02rem;font-weight:700;display:block;line-height:1.15}
header .brand span{font-size:.7rem;color:var(--muted)}
header .right{display:flex;align-items:center;gap:14px}
#clock{font-size:.82rem;color:var(--muted);font-variant-numeric:tabular-nums}
.live{display:flex;align-items:center;gap:7px;background:#eafbf0;color:#166534;padding:6px 12px;border-radius:20px;font-weight:600;font-size:.74rem}
.live .dot{width:8px;height:8px;border-radius:50%;background:#22c55e;animation:pulse 1.6s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(34,197,94,.5)}70%{box-shadow:0 0 0 7px rgba(34,197,94,0)}100%{box-shadow:0 0 0 0 rgba(34,197,94,0)}}
.wrap{max-width:1640px;margin:18px auto;padding:0 24px}
.rowg{display:grid;gap:16px;margin-bottom:16px}
.kpis{grid-template-columns:repeat(4,1fr)}
.c2{grid-template-columns:1fr 1fr}
.masonry{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}
.col{display:flex;flex-direction:column;gap:16px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);overflow:hidden}
.panel .ph{padding:14px 18px;border-bottom:1px solid var(--line);font-size:.86rem;font-weight:600;display:flex;align-items:center;gap:8px}
.panel .pb{padding:16px 18px}
.panel.full{grid-column:1 / -1}
/* KPI */
.kpi{display:flex;align-items:center;gap:14px;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;box-shadow:var(--shadow)}
.kpi .ic{width:46px;height:46px;border-radius:12px;display:grid;place-items:center;font-size:21px;flex:none}
.kpi .num{font-size:1.75rem;font-weight:700;line-height:1}
.kpi .lbl{font-size:.78rem;color:var(--muted);margin-top:4px}
.kpi.g .ic{background:#dcfce7}.kpi.b .ic{background:#dbeafe}.kpi.p .ic{background:#ede9fe}.kpi.a .ic{background:#fef3c7}
/* progress */
.bar{margin-bottom:16px}.bar:last-child{margin-bottom:0}
.bar .t{display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:6px}
.bar .t b{font-weight:600}.bar .t span{color:var(--muted);font-variant-numeric:tabular-nums}
.track{height:9px;background:#eef1f6;border-radius:6px;overflow:hidden}
.fill{height:100%;border-radius:6px;transition:width .5s}
/* donut */
.donut-wrap{display:flex;align-items:center;gap:24px}
.donut{position:relative;width:148px;height:148px;flex:none}
.donut .ctr{position:absolute;inset:0;display:grid;place-items:center;text-align:center}
.donut .ctr b{font-size:1.7rem;font-weight:700}
.donut .ctr small{font-size:.7rem;color:var(--muted)}
.legend div{display:flex;align-items:center;gap:9px;font-size:.83rem;margin-bottom:11px}
.legend div:last-child{margin-bottom:0}
.legend .sw{width:12px;height:12px;border-radius:3px}
.legend b{font-variant-numeric:tabular-nums}
/* tables */
table{width:100%;border-collapse:collapse;font-size:.83rem}
th{text-align:left;color:var(--muted);font-weight:600;font-size:.7rem;text-transform:uppercase;letter-spacing:.4px;padding:9px 12px;border-bottom:1px solid var(--line)}
td{padding:10px 12px;border-bottom:1px solid #f1f3f5}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:#f8fafc}
.pill{display:inline-block;padding:3px 11px;border-radius:20px;font-size:.72rem;font-weight:600;white-space:nowrap}
.pill.g{background:#dcfce7;color:#166534}.pill.a{background:#fef3c7;color:#92400e}.pill.r{background:#fee2e2;color:#991b1b}.pill.b{background:#dbeafe;color:#1e40af}
.score{font-variant-numeric:tabular-nums;font-weight:600}
.presence{display:flex;flex-wrap:wrap;gap:12px}
.pcard{display:flex;align-items:center;gap:12px;background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:11px 15px;min-width:200px}
.pca{width:40px;height:40px;border-radius:50%;background:linear-gradient(135deg,#22c55e,#16a34a);color:#fff;display:grid;place-items:center;font-weight:700;box-shadow:0 0 0 3px #dcfce7;flex:none}
.pcard .nm{font-weight:600}.pcard .zn{font-size:.74rem;color:var(--muted);margin-top:1px}.pcard .tm{font-size:.71rem;color:#9ca3af;margin-top:1px}
.empty{color:#9ca3af;padding:20px;text-align:center;font-size:.85rem;display:block}
@media(max-width:980px){.kpis{grid-template-columns:1fr 1fr}.c2{grid-template-columns:1fr}.masonry{grid-template-columns:1fr}}
</style></head>
<body>
<header><div class="in">
  <div class="brand"><div class="logo">🏭</div><div><b>Warehouse Access Monitor</b><span>Giám sát truy cập &amp; hiện diện — RFID + Computer Vision</span></div></div>
  <div class="right"><span id="clock"></span><span class="live"><span class="dot"></span>LIVE</span></div>
</div></header>

<main class="wrap">
  <div class="rowg kpis">
    <div class="kpi g"><div class="ic">👥</div><div><div class="num" id="k_pres">0</div><div class="lbl">Đang trong kho</div></div></div>
    <div class="kpi b"><div class="ic">🚪</div><div><div class="num" id="k_ent">0</div><div class="lbl">Lượt vào cửa hôm nay</div></div></div>
    <div class="kpi p"><div class="ic">💳</div><div><div class="num" id="k_rfid">0</div><div class="lbl">Lượt quẹt thẻ</div></div></div>
    <div class="kpi a"><div class="ic">⚠️</div><div><div class="num" id="k_anom">0</div><div class="lbl">Cảnh báo bất thường</div></div></div>
  </div>

  <div class="rowg c2">
    <div class="panel"><div class="ph">🔐 Kết quả xác thực</div>
      <div class="pb"><div class="donut-wrap">
        <div class="donut">
          <svg viewBox="0 0 42 42" width="148" height="148" style="transform:rotate(-90deg)">
            <circle cx="21" cy="21" r="15.915" fill="none" stroke="#eef1f6" stroke-width="5"></circle>
            <circle id="dseg" cx="21" cy="21" r="15.915" fill="none" stroke="#16a34a" stroke-width="5" stroke-dasharray="0 100" stroke-linecap="round"></circle>
          </svg>
          <div class="ctr"><div><b id="dpct">0%</b><small>hợp lệ</small></div></div>
        </div>
        <div class="legend">
          <div><span class="sw" style="background:#16a34a"></span>Hợp lệ&nbsp; <b id="lg_ok">0</b></div>
          <div><span class="sw" style="background:#d97706"></span>Bất thường&nbsp; <b id="lg_ab">0</b></div>
          <div><span class="sw" style="background:#eef1f6;border:1px solid #e5e7eb"></span>Tổng sự kiện&nbsp; <b id="lg_tt">0</b></div>
        </div>
      </div></div>
    </div>
    <div class="panel"><div class="ph">🎯 Tổng quan hệ thống</div>
      <div class="pb">
        <div class="bar"><div class="t"><b>Điểm danh hôm nay</b><span id="b1t">0</span></div><div class="track"><div class="fill" id="b1" style="width:0;background:#2563eb"></div></div></div>
        <div class="bar"><div class="t"><b>Thẻ hợp lệ (RFID)</b><span id="b2t">0</span></div><div class="track"><div class="fill" id="b2" style="width:0;background:#16a34a"></div></div></div>
        <div class="bar"><div class="t"><b>Tỉ lệ bất thường</b><span id="b3t">0</span></div><div class="track"><div class="fill" id="b3" style="width:0;background:#dc2626"></div></div></div>
      </div>
    </div>
  </div>

  <div class="rowg c2">
    <div class="panel"><div class="ph">👥 Đang có mặt trong kho (thời gian thực)</div>
      <div class="pb"><div class="presence" id="pbox"><span class="empty">Chưa có người nào trong kho…</span></div></div>
    </div>
    <div class="panel"><div class="ph">🚪 Vào cửa hôm nay</div>
      <div class="pb" style="padding:4px 6px"><table id="t1"><thead><tr><th>Nhân viên</th><th>Giờ vào</th><th>Zone</th><th>Score</th></tr></thead><tbody></tbody></table></div></div>
  </div>

  <div class="rowg c2">
    <div class="panel"><div class="ph">⚠️ Cảnh báo bất thường</div>
      <div class="pb" style="padding:4px 6px"><table id="t3"><thead><tr><th>Loại</th><th>Nhân viên</th><th>Chi tiết</th><th>Thời gian</th></tr></thead><tbody></tbody></table></div></div>
    <div class="panel"><div class="ph">💳 Sự kiện RFID</div>
      <div class="pb" style="padding:4px 6px"><table id="t2"><thead><tr><th>UID</th><th>Nhân viên</th><th>Thời gian</th></tr></thead><tbody></tbody></table></div></div>
  </div>
</main>
<script>
const NAMES={
  "NV001":"Bo Man","NV002":"Me Mai","NV003":"Anh Minh",
  "NV004":"Chi Dung","NV005":"Toi"
}
const VI_TEXT={
  "no_rfid_intruder":"Phát hiện người đi qua nhưng không có RFID",
  "unknown_uid":"UID thẻ không có trong hệ thống",
  "proxy_swipe":"Nghi ngờ quẹt thẻ hộ",
  "visual_mismatch":"Thẻ và dáng người không khớp",
  "visual_mismatch_low_confidence":"Khuôn mặt phát hiện nhưng độ tin cậy thấp",
  "rfid_no_crossing":"Thẻ đã quẹt nhưng không phát hiện người đi qua cửa",
  "no_face_at_gate":"Không thấy khuôn mặt tại cổng khi quẹt thẻ",
  "unknown_person":"Phát hiện người chưa xác định",
  "rfid_face_verified":"Xác nhận bằng khuôn mặt (RFID-trigger)",
  "rfid_presence_only_fallback":"Xác nhận bằng hiện diện tại cửa",
  "rfid_body_fallback":"Xác nhận bằng dáng người (RFID-trigger)",
  "rfid_cv_fusion":"Xác nhận bằng RFID + CV",
  "cv_zone2":"Re-ID Zone 2",
  "cv_zone2_face_match":"Khớp khuôn mặt Zone 2"
}
const SEV={
  "no_rfid_intruder":"r","unknown_uid":"r","proxy_swipe":"r","unknown_person":"r",
  "visual_mismatch":"a","visual_mismatch_low_confidence":"a",
  "rfid_no_crossing":"a","no_face_at_gate":"a"
}
const TOTAL_EMP=Object.keys(NAMES).length
function name(id){return id?(NAMES[id]||id)+' ('+id+')':''}
function fmt(s){return s?s.replace('T',' ').substring(0,19):''}
function initials(id){const n=(NAMES[id]||id||'?').trim();return n.split(/\\s+/).map(w=>w[0]).slice(0,2).join('').toUpperCase()}
function tick(){const d=new Date();document.getElementById('clock').textContent=d.toLocaleTimeString('vi-VN')+' · '+d.toLocaleDateString('vi-VN')}
setInterval(tick,1000);tick()
function setBar(id,pct,txt){document.getElementById(id).style.width=Math.max(0,Math.min(100,pct))+'%';document.getElementById(id+'t').textContent=txt}
async function refresh(){
  const [e,a,r,p]=await Promise.all([
    fetch('/api/entries').then(x=>x.json()),
    fetch('/api/anomalies').then(x=>x.json()),
    fetch('/api/rfid').then(x=>x.json()),
    fetch('/api/presence').then(x=>x.json())
  ])

  document.getElementById('k_pres').textContent=p.length
  document.getElementById('k_ent').textContent=e.length
  document.getElementById('k_rfid').textContent=r.length
  document.getElementById('k_anom').textContent=a.length

  const entered=new Set(e.map(x=>x.employee_id)).size
  const validRfid=r.filter(x=>x.employee_id).length
  const totRfid=r.length
  const abnRatio=(e.length+a.length)?Math.round(a.length/(e.length+a.length)*100):0
  setBar('b1',TOTAL_EMP?entered/TOTAL_EMP*100:0,entered+'/'+TOTAL_EMP)
  setBar('b2',totRfid?validRfid/totRfid*100:0,validRfid+'/'+totRfid)
  setBar('b3',abnRatio,abnRatio+'%')

  const ok=e.length, ab=a.length, tt=ok+ab
  const pct=tt?Math.round(ok/tt*100):0
  document.getElementById('dseg').setAttribute('stroke-dasharray',pct+' '+(100-pct))
  document.getElementById('dpct').textContent=pct+'%'
  document.getElementById('lg_ok').textContent=ok
  document.getElementById('lg_ab').textContent=ab
  document.getElementById('lg_tt').textContent=tt

  const pb=document.getElementById('pbox')
  pb.innerHTML = p.length ? p.map(x=>`
    <div class="pcard">
      <div class="pca">${initials(x.employee_id)}</div>
      <div>
        <div class="nm">${NAMES[x.employee_id]||x.employee_id}</div>
        <div class="zn">Zone ${x.current_zone ?? x.zone} · ${x.track_key||x.track_id||''}</div>
        <div class="tm">${fmt(x.last_seen)}</div>
      </div>
    </div>`).join('') : '<span class="empty">Chưa có người nào trong kho…</span>'

  document.querySelector('#t1 tbody').innerHTML=e.map(x=>
    `<tr><td><span class="pill g">${name(x.employee_id)}</span></td><td>${fmt(x.entry_time)}</td><td>Zone ${x.zone}</td><td class="score">${(x.fusion_score||0).toFixed(3)}</td></tr>`
  ).join('')||'<tr><td colspan=4 class="empty">Chưa có dữ liệu</td></tr>'

  document.querySelector('#t2 tbody').innerHTML=r.map(x=>
    `<tr><td>${x.uid}</td><td>${x.employee_id?`<span class="pill b">${name(x.employee_id)}</span>`:'<span class="pill r">unknown</span>'}</td><td>${fmt(x.timestamp)}</td></tr>`
  ).join('')||'<tr><td colspan=3 class="empty">Chưa có dữ liệu</td></tr>'

  document.querySelector('#t3 tbody').innerHTML=a.map(x=>
    `<tr><td><span class="pill ${SEV[x.type]||'a'}">${VI_TEXT[x.type]||x.type}</span></td><td>${name(x.employee_id)}</td><td>${x.detail||''}</td><td>${fmt(x.timestamp)}</td></tr>`
  ).join('')||'<tr><td colspan=4 class="empty">Không có cảnh báo</td></tr>'
}
refresh();setInterval(refresh,3000)
</script></body></html>"""

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

def run_dashboard(db):
    # Tắt log truy cập HTTP của Flask (mỗi 3s dashboard poll 4 API) cho terminal sạch khi quay demo.
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    init_dashboard(db)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG, use_reloader=False)
