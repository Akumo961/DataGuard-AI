(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  let accessToken = "";
  let lastResult = null;
  const titles = {overview:"Vue d'ensemble",discovery:"Découverte",findings:"Constats PII",pia:"Évaluations PIA",remediation:"Remédiation",audit:"Audit & preuves"};

  document.querySelectorAll(".nav").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll(".nav").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    button.classList.add("active"); const view = $(button.dataset.view); view.classList.add("active"); $("page-title").textContent = titles[button.dataset.view];
  }));

  $("tokenBtn").addEventListener("click", () => { $("token").value = ""; $("tokenDialog").showModal(); });
  $("saveToken").addEventListener("click", () => { accessToken = $("token").value.trim(); $("message").textContent = accessToken ? "Jeton chargé en mémoire pour cette page." : "Aucun jeton configuré."; });

  function render(result) {
    lastResult = result;
    $("globalScore").textContent = Number(result.risk.score).toFixed(1);
    $("riskLevel").textContent = result.risk.level;
    $("piiCount").textContent = result.detections.length;
    $("critical").textContent = result.risk.level === "CRITICAL" ? "1" : "0";
    $("confidence").textContent = result.detections.length ? `${(result.detections.reduce((s,d)=>s+d.confidence,0)/result.detections.length*100).toFixed(0)}%` : "—";
    $("factors").innerHTML = result.risk.factors.map(f => `<div class="factor"><b>${escapeHtml(f.name)} · +${Number(f.points).toFixed(1)}</b><small>${escapeHtml(f.detail)}</small></div>`).join("") || '<p class="empty">Aucun facteur.</p>';
    $("findingsBody").innerHTML = result.detections.map(d => `<tr><td>${escapeHtml(d.type)}</td><td>${d.start}–${d.end}</td><td class="confidence">${(d.confidence*100).toFixed(0)}%</td><td>${escapeHtml(d.detector)}</td><td>${escapeHtml(d.redacted_value)}</td></tr>`).join("") || '<tr><td colspan="5" class="empty">Aucune détection.</td></tr>';
  }
  function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c])); }

  $("analyze").addEventListener("click", async () => {
    const text = $("input").value.trim(); if (!text) { $("message").textContent = "Entrez un contenu à analyser."; return; }
    if (!accessToken) { $("message").textContent = "Un Bearer token valide est requis par l'API."; return; }
    const button = $("analyze"); button.disabled = true; $("message").textContent = "Analyse en cours…";
    try {
      const response = await fetch("/api/v1/analyze", { method:"POST", headers:{"Content-Type":"application/json","Authorization":`Bearer ${accessToken}`}, body:JSON.stringify({text,access_scope:$("access").value,exposure:$("exposure").value,encrypted_at_rest:$("encrypted").value === "true",purpose_defined:true,data_location:"canada",retention_days:365,framework:"quebec_privacy"}) });
      const data = await response.json(); if (!response.ok) throw new Error(data.detail || `Erreur API (${response.status})`); render(data); $("message").textContent = "Analyse terminée. Les valeurs sensibles sont redigées par l'API.";
    } catch (error) { $("message").textContent = error instanceof Error ? error.message : "Échec de l'analyse."; }
    finally { button.disabled = false; }
  });

  window.addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") $("analyze").click(); });
})();