(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  let accessToken = "";
  const titles = {
    overview: "Vue d'ensemble",
    discovery: "Découverte",
    findings: "Constats PII",
    pia: "Évaluations PIA",
    remediation: "Remédiation",
    audit: "Audit & preuves",
  };

  document.querySelectorAll(".nav").forEach((button) =>
    button.addEventListener("click", async () => {
      document.querySelectorAll(".nav").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
      button.classList.add("active");
      const view = $(button.dataset.view);
      view.classList.add("active");
      $("page-title").textContent = titles[button.dataset.view];
      if (button.dataset.view === "findings") await loadFindings();
      if (button.dataset.view === "pia") await loadPias();
      if (button.dataset.view === "remediation") await loadRemediations();
    }),
  );

  $("loginBtn").addEventListener("click", () => {
    $("loginMessage").textContent = "";
    $("loginDialog").showModal();
  });

  $("loginSubmit").addEventListener("click", async (event) => {
    event.preventDefault();
    const button = $("loginSubmit");
    button.disabled = true;
    $("loginMessage").textContent = "Connexion en cours…";
    try {
      const response = await fetch("/api/v1/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          organization_slug: $("orgSlug").value.trim(),
          email: $("email").value.trim(),
          password: $("password").value,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Erreur de connexion (${response.status})`);
      accessToken = data.access_token;
      $("sessionStatus").textContent = "Connecté";
      $("loginBtn").hidden = true;
      $("logoutBtn").hidden = false;
      $("loginDialog").close();
      $("message").textContent = "Session authentifiée en mémoire.";
    } catch (error) {
      $("loginMessage").textContent = error instanceof Error ? error.message : "Échec de connexion.";
    } finally {
      button.disabled = false;
    }
  });

  $("logoutBtn").addEventListener("click", async () => {
    if (!accessToken) return;
    try {
      await apiFetch("/api/v1/auth/logout", { method: "POST" });
    } finally {
      accessToken = "";
      $("sessionStatus").textContent = "Non connecté";
      $("loginBtn").hidden = false;
      $("logoutBtn").hidden = true;
      $("message").textContent = "Session terminée.";
    }
  });

  function apiFetch(url, options = {}) {
    const headers = new Headers(options.headers || {});
    headers.set("Authorization", `Bearer ${accessToken}`);
    return fetch(url, { ...options, headers });
  }

  function render(result) {
    $("globalScore").textContent = Number(result.risk.score).toFixed(1);
    $("riskLevel").textContent = result.risk.level;
    $("piiCount").textContent = result.detections.length;
    $("critical").textContent = result.risk.level === "CRITICAL" ? "1" : "0";
    $("confidence").textContent = result.detections.length
      ? `${(result.detections.reduce((sum, item) => sum + item.confidence, 0) / result.detections.length * 100).toFixed(0)}%`
      : "—";
    $("factors").innerHTML = result.risk.factors.map((factor) =>
      `<div class="factor"><b>${escapeHtml(factor.name)} · +${Number(factor.points).toFixed(1)}</b><small>${escapeHtml(factor.detail)}</small></div>`,
    ).join("") || '<p class="empty">Aucun facteur.</p>';
    $("findingsBody").innerHTML = result.detections.map((detection) =>
      `<tr><td>${escapeHtml(detection.type)}</td><td>${detection.start}–${detection.end}</td><td class="confidence">${(detection.confidence * 100).toFixed(0)}%</td><td>${escapeHtml(detection.detector)}</td><td>${escapeHtml(detection.redacted_value)}</td></tr>`,
    ).join("") || '<tr><td colspan="5" class="empty">Aucune détection.</td></tr>';
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>'"]/g, (char) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "'": "&#39;",
      '"': "&quot;",
    }[char]));
  }

  async function loadFindings() {
    if (!accessToken) return;
    try {
      const response = await apiFetch("/api/v1/findings?limit=100");
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Impossible de charger les constats.");
      $("findingsBody").innerHTML = data.map((finding) =>
        `<tr><td>${escapeHtml(finding.pii_type)}</td><td>${finding.start_offset}–${finding.end_offset}</td><td>${(finding.confidence * 100).toFixed(0)}%</td><td>${escapeHtml(finding.detector)}</td><td>${escapeHtml(finding.evidence.redacted_value || "[REDACTED]")}</td></tr>`,
      ).join("") || '<tr><td colspan="5" class="empty">Aucun constat.</td></tr>';
    } catch (error) {
      $("message").textContent = error instanceof Error ? error.message : "Échec de chargement.";
    }
  }

  async function loadPias() {
    if (!accessToken) return;
    try {
      const response = await apiFetch("/api/v1/pias?limit=100");
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Impossible de charger les PIA.");
      $("piaCount").textContent = data.length;
      $("piaOpen").textContent = data.filter((pia) => pia.status !== "APPROVED" && pia.status !== "ARCHIVED").length;
      $("piaApproved").textContent = data.filter((pia) => pia.status === "APPROVED").length;
      $("piaBody").innerHTML = data.map((pia) =>
        `<tr><td>${escapeHtml(pia.project_name)}</td><td>${escapeHtml(pia.status)}</td><td>${pia.version}</td></tr>`,
      ).join("") || '<tr><td colspan="3" class="empty">Aucune PIA.</td></tr>';
    } catch (error) {
      $("message").textContent = error instanceof Error ? error.message : "Échec de chargement.";
    }
  }

  async function loadRemediations() {
    if (!accessToken) return;
    try {
      const response = await apiFetch("/api/v1/remediations?limit=100");
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Impossible de charger les remédiations.");
      $("remediationBody").innerHTML = data.map((item) =>
        `<tr><td>${escapeHtml(item.status)}</td><td>${escapeHtml(item.priority)}</td><td>${escapeHtml(item.owner_id || "—")}</td><td>${escapeHtml(item.due_at || "—")}</td></tr>`,
      ).join("") || '<tr><td colspan="4" class="empty">Aucune tâche.</td></tr>';
    } catch (error) {
      $("message").textContent = error instanceof Error ? error.message : "Échec de chargement.";
    }
  }

  $("refreshFindings").addEventListener("click", loadFindings);
  $("refreshPia").addEventListener("click", loadPias);
  $("refreshRemediation").addEventListener("click", loadRemediations);

  $("analyze").addEventListener("click", async () => {
    const text = $("input").value.trim();
    if (!text) {
      $("message").textContent = "Entrez un contenu à analyser.";
      return;
    }
    if (!accessToken) {
      $("message").textContent = "Connectez-vous pour analyser un contenu.";
      return;
    }
    const button = $("analyze");
    button.disabled = true;
    $("message").textContent = "Analyse en cours…";
    try {
      const response = await apiFetch("/api/v1/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          access_scope: $("access").value,
          exposure: $("exposure").value,
          encrypted_at_rest: $("encrypted").value === "true",
          purpose_defined: true,
          data_location: "canada",
          retention_days: 365,
          framework: "quebec_privacy",
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Erreur API (${response.status})`);
      render(data);
      $("message").textContent = "Analyse terminée. Les valeurs sensibles sont redigées par l'API.";
      await loadFindings();
    } catch (error) {
      $("message").textContent = error instanceof Error ? error.message : "Échec de l'analyse.";
    } finally {
      button.disabled = false;
    }
  });

  window.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") $("analyze").click();
  });
})();
