"""داشبورد تعاملی تک‌صفحه‌ای (FR-07). بدون منبع خارجی؛ اسکریپت با nonce سرور اجرا می‌شود.

همه متن‌های دریافتی از سرور با ``textContent`` درج می‌شوند (نه innerHTML) تا XSS ممکن نباشد.
"""

DASHBOARD_HTML = """<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IPIND² — طراحی هوشمند نانوحامل</title>
<style nonce="__NONCE__">
:root{--bg:#f6f8fa;--card:#fff;--ink:#1a1a1a;--muted:#5b6670;--accent:#0b5cad;--bad:#b3261e;--ok:#1b7f3b;--line:#d8dee4}
@media (prefers-color-scheme: dark){:root{--bg:#10151a;--card:#182027;--ink:#e8edf2;--muted:#9aa7b2;--accent:#6aaeea;--line:#2b3640}}
*{box-sizing:border-box}body{margin:0;font-family:Tahoma,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{padding:1rem 1.5rem;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}
main{max-width:1100px;margin:0 auto;padding:1rem 1.5rem 3rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:1rem;margin:1rem 0}
label{display:block;margin:.5rem 0 .2rem;color:var(--muted);font-size:.9rem}
input,textarea,select{width:100%;padding:.5rem;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);font:inherit}
button{background:var(--accent);color:#fff;border:0;border-radius:6px;padding:.55rem 1.1rem;font:inherit;cursor:pointer;margin-top:.7rem}
button.secondary{background:transparent;color:var(--accent);border:1px solid var(--accent)}
button:disabled{opacity:.5;cursor:wait}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.8rem}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border:1px solid var(--line);padding:.35rem .5rem;text-align:right}
code{direction:ltr;unicode-bidi:embed;word-break:break-all;font-size:.82em}
.msg{padding:.6rem;border-radius:6px;margin:.5rem 0}.msg.err{background:color-mix(in srgb,var(--bad) 15%,transparent);color:var(--bad)}
.msg.warn{background:color-mix(in srgb,#e0b84a 22%,transparent)}.msg.ok{background:color-mix(in srgb,var(--ok) 15%,transparent);color:var(--ok)}
.hidden{display:none}.muted{color:var(--muted)}
.bar{height:10px;background:var(--line);border-radius:5px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--accent)}
@media (max-width:600px){main{padding:.5rem}}
</style>
</head>
<body>
<header><strong>IPIND² · طراحی هوشمند نانوحامل</strong><span id="who" class="muted"></span></header>
<main>
<section id="login" class="card">
  <h2>ورود</h2>
  <div class="row">
    <div><label for="u">نام کاربری</label><input id="u" autocomplete="username"></div>
    <div><label for="p">رمز عبور</label><input id="p" type="password" autocomplete="current-password"></div>
    <div><label for="t">کد دومرحله‌ای (TOTP)</label><input id="t" inputmode="numeric" maxlength="8" autocomplete="one-time-code"></div>
  </div>
  <button id="loginBtn">ورود</button><div id="loginMsg"></div>
</section>

<div id="app" class="hidden">
  <section class="card">
    <h2>طراحی (پرس‌وجوی زبان طبیعی)</h2>
    <label for="q">توضیح نانوحامل مورد نظر</label>
    <textarea id="q" rows="3" maxlength="1000">یک نانوحامل لیپیدی برای هدف‌گیری تومور، اندازه بین ۸۰ تا ۱۲۰ نانومتر</textarea>
    <div class="row">
      <div><label for="ng">تعداد ساختار تولیدی</label><input id="ng" type="number" value="1500" min="50" max="20000"></div>
      <div><label for="nf">کاندیداهای نهایی</label><input id="nf" type="number" value="5" min="1" max="10"></div>
    </div>
    <button id="parseBtn" class="secondary">پیش‌نمایش پارامترهای استخراج‌شده</button>
    <button id="designBtn">اجرای طراحی</button>
    <div id="parsed" class="muted"></div><div id="designMsg"></div>
  </section>
  <section id="results" class="card hidden"><h2>نتایج</h2><div id="resultBody"></div>
    <button id="reportBtn" class="secondary">گزارش HTML</button>
    <button id="jsonldBtn" class="secondary">خروجی FAIR (JSON-LD)</button></section>
  <section class="card">
    <h2>نتایج آزمایشگاهی (CSV)</h2>
    <label for="csv">ستون‌ها: smiles,experimental_size_nm,experimental_zeta_potential,experimental_loading_efficiency,experimental_cytotoxicity,experimental_date,lab_technician</label>
    <textarea id="csv" rows="4"></textarea>
    <button id="labBtn">ثبت نتایج</button><button id="alBtn" class="secondary">به‌روزرسانی مدل از بازخورد</button><div id="labMsg"></div>
  </section>
</div>
</main>
<script nonce="__NONCE__">
(function(){
  var token=null, lastJob=null;
  var $=function(id){return document.getElementById(id)};
  function msg(el,text,kind){el.textContent="";var d=document.createElement("div");d.className="msg "+(kind||"");d.textContent=text;el.appendChild(d)}
  function api(method,path,body){
    var h={"Content-Type":"application/json"}; if(token) h["Authorization"]="Bearer "+token;
    return fetch(path,{method:method,headers:h,body:body?JSON.stringify(body):undefined}).then(function(r){
      return r.json().catch(function(){return {}}).then(function(j){ if(!r.ok) throw new Error(j.detail||("خطا "+r.status)); return j; });
    });
  }
  function cell(tr,text,isCode){var td=document.createElement("td"); if(isCode){var c=document.createElement("code");c.textContent=text;td.appendChild(c)}else td.textContent=text; tr.appendChild(td)}
  function fmt(v){return typeof v==="number"?(Math.abs(v)<10?v.toFixed(3):v.toFixed(1)):String(v)}

  $("loginBtn").onclick=function(){
    api("POST","/auth/login",{username:$("u").value,password:$("p").value,totp_code:$("t").value}).then(function(j){
      token=j.access_token; $("login").classList.add("hidden"); $("app").classList.remove("hidden"); $("who").textContent=$("u").value; $("p").value=""; $("t").value="";
    }).catch(function(e){msg($("loginMsg"),e.message,"err")});
  };
  $("parseBtn").onclick=function(){
    api("POST","/nlp/parse",{query:$("q").value}).then(function(j){ $("parsed").textContent=JSON.stringify(j,null,1); }).catch(function(e){msg($("parsed"),e.message,"err")});
  };
  function poll(id){
    return api("GET","/jobs/"+id).then(function(j){
      if(j.status==="done") return j; if(j.status==="failed") throw new Error(j.error||"کار ناموفق بود");
      return new Promise(function(r){setTimeout(r,1500)}).then(function(){return poll(id)});
    });
  }
  function render(result){
    var body=$("resultBody"); body.textContent="";
    (result.warnings||[]).forEach(function(w){var d=document.createElement("div");d.className="msg warn";d.textContent="⚠ "+w;body.appendChild(d)});
    var t=document.createElement("table"); var head=document.createElement("tr");
    ["رتبه","SMILES","اندازه nm","زتا mV","بارگذاری ٪","IC50","اطمینان","MD"].forEach(function(h){var th=document.createElement("th");th.textContent=h;head.appendChild(th)}); t.appendChild(head);
    (result.final_candidates||[]).forEach(function(c){
      var tr=document.createElement("tr"), p=c.predictions;
      cell(tr,String(c.rank)); cell(tr,c.smiles,true);
      cell(tr,fmt(p.phys_size_nm)); cell(tr,fmt(p.phys_zeta_potential_mV)); cell(tr,fmt(p.phys_drug_loading_efficiency_percent)); cell(tr,fmt(p.bio_cytotoxicity_ic50_ug_ml));
      cell(tr,c.overall_confidence==null?"—":fmt(c.overall_confidence)); cell(tr,c.md_stable==null?"—":(c.md_stable?"پایدار":"ناپایدار")); t.appendChild(tr);
    });
    body.appendChild(t); $("results").classList.remove("hidden");
  }
  $("designBtn").onclick=function(){
    var b=$("designBtn"); b.disabled=true; msg($("designMsg"),"در حال اجرا… (ممکن است چند دقیقه طول بکشد)","");
    api("POST","/design",{query:$("q").value,n_generate:parseInt($("ng").value,10),n_final:parseInt($("nf").value,10)})
      .then(function(j){lastJob=j.job_id; return poll(j.job_id)})
      .then(function(j){msg($("designMsg"),"انجام شد","ok"); render(j.result)})
      .catch(function(e){msg($("designMsg"),e.message,"err")}).then(function(){b.disabled=false});
  };
  function download(path,name){
    fetch(path,{headers:{"Authorization":"Bearer "+token}}).then(function(r){return r.blob()}).then(function(blob){
      var a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=name; a.click(); URL.revokeObjectURL(a.href);
    });
  }
  $("reportBtn").onclick=function(){ if(lastJob) download("/reports/"+lastJob,"report.html") };
  $("jsonldBtn").onclick=function(){ if(lastJob) download("/export/"+lastJob+"?fmt=jsonld","ipind2.jsonld") };
  $("labBtn").onclick=function(){
    var lines=$("csv").value.trim().split("\\n"); if(lines.length<2) return msg($("labMsg"),"CSV خالی است","err");
    var cols=lines[0].split(",").map(function(s){return s.trim()});
    var rows=lines.slice(1).map(function(l){var v=l.split(","),o={};cols.forEach(function(c,i){o[c]=(v[i]||"").trim()||null});return o});
    api("POST","/lab/results",{rows:rows}).then(function(j){msg($("labMsg"),j.accepted+" سطر ثبت شد؛ در انتظار به‌روزرسانی: "+j.pending,"ok")}).catch(function(e){msg($("labMsg"),e.message,"err")});
  };
  $("alBtn").onclick=function(){
    api("POST","/active-learning/retrain",{force:false}).then(function(j){msg($("labMsg"),j.message,j.retrained?"ok":"warn")}).catch(function(e){msg($("labMsg"),e.message,"err")});
  };
})();
</script>
</body>
</html>
"""


def render_dashboard(nonce: str) -> str:
    return DASHBOARD_HTML.replace("__NONCE__", nonce)
