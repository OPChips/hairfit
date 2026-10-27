/* HairFit · 髮型试镜 — vanilla ES module, no build step */

const $ = (sel) => document.querySelector(sel);

const state = {
  file: null,       // 原始 File，generate 时二次上传
  rec: null,        // /v1/analyze 完整响应
  jobId: null,
  styles: {},       // id -> {name, ref_image}
  pollTimer: null,
};

const FACE_SHAPE = {
  oval: "鹅蛋形", round: "圆形", square: "方形", heart: "心形",
  diamond: "菱形", oblong: "长形", triangle: "三角形",
};
const VOLUME = { thin: "偏少", medium: "中等", thick: "浓密" };
const HAIRLINE = { normal: "正常", high: "偏高", receding: "后移" };

// ---------------------------------------------------------------------------
// 步骤切换
// ---------------------------------------------------------------------------

function setStep(n) {
  for (const el of document.querySelectorAll(".chapter")) {
    const k = Number(el.dataset.step);
    el.classList.toggle("active", k === n);
    el.classList.toggle("done", k < n);
  }
  for (const el of document.querySelectorAll(".act")) el.hidden = true;
  const act = $(`#step${n}`);
  act.hidden = false;
  // 重新触发入场动画
  act.style.animation = "none";
  void act.offsetWidth;
  act.style.animation = "";
  hideError();
  $("#report-wrap").hidden = true;
}

function showError(msg) {
  const el = $("#error");
  el.textContent = msg;
  el.hidden = false;
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
function hideError() { $("#error").hidden = true; }

// ---------------------------------------------------------------------------
// 壹 · 上传 + 分析
// ---------------------------------------------------------------------------

$("#photo-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  state.file = file;
  $("#preview").src = URL.createObjectURL(file);
  $("#upload-area").hidden = true;
  $("#preview-wrap").hidden = false;
});

function resetPhoto() {
  state.file = null;
  $("#photo-input").value = "";
  $("#preview").src = "";
  $("#upload-area").hidden = false;
  $("#preview-wrap").hidden = true;
}

$("#btn-reanalyze").addEventListener("click", resetPhoto);
$("#btn-back1").addEventListener("click", () => { resetPhoto(); setStep(1); });
$("#btn-back2").addEventListener("click", () => setStep(2));

$("#btn-analyze").addEventListener("click", async () => {
  if (!state.file) return;
  const btn = $("#btn-analyze");
  btn.disabled = true;
  $("#analyze-status").hidden = false;
  try {
    const form = new FormData();
    form.append("photo", state.file);
    const resp = await fetch("/v1/analyze", { method: "POST", body: form });
    if (!resp.ok) throw new Error(await detail(resp));
    state.rec = await resp.json();
    renderStep2();
    setStep(2);
  } catch (err) {
    showError(`分析失败：${err.message}`);
  } finally {
    btn.disabled = false;
    $("#analyze-status").hidden = true;
  }
});

async function detail(resp) {
  try {
    const body = await resp.json();
    return body.detail || JSON.stringify(body);
  } catch { return `HTTP ${resp.status}`; }
}

// ---------------------------------------------------------------------------
// 贰 · 脸部档案 + 选款
// ---------------------------------------------------------------------------

async function loadStyles() {
  if (Object.keys(state.styles).length) return;
  try {
    const resp = await fetch("/v1/styles");
    if (resp.ok) {
      for (const s of await resp.json()) state.styles[s.id] = s;
    }
  } catch { /* 名称缺失不阻塞主流程 */ }
}

function styleName(id) { return state.styles[id]?.name ?? id; }

function renderStep2() {
  const fa = state.rec.face_analysis;
  const fields = [
    ["脸型", FACE_SHAPE[fa.face_shape] ?? fa.face_shape],
    ["比例", fa.face_ratio],
    ["发量", VOLUME[fa.hair_volume] ?? fa.hair_volume],
    ["发质", fa.hair_texture],
    ["发际线", HAIRLINE[fa.hairline] ?? fa.hairline],
  ];
  $("#face-card").innerHTML = `
    <div class="dossier-tags">
      ${fields.map(([k, v]) => `<span class="field"><span class="k">${k}</span><span class="v">${v}</span></span>`).join("")}
    </div>
    ${fa.notes ? `<p class="dossier-notes">${fa.notes}</p>` : ""}`;

  $("#rec-count").textContent = ` ${state.rec.recommended.length} `;

  const list = $("#style-list");
  list.innerHTML = "";
  state.rec.recommended.forEach((r, i) => {
    const meta = state.styles[r.id];
    const label = document.createElement("label");
    label.className = "look";
    label.innerHTML = `
      <span class="num">LOOK ${String(i + 1).padStart(2, "0")}</span>
      ${meta?.ref_image ? `<img src="${meta.ref_image}" alt="${r.id}">` : ""}
      <span class="info">
        <span class="name">${styleName(r.id)}</span>
        <span class="reason">${r.reason}</span>
      </span>
      <input type="checkbox" value="${r.id}" checked>`;
    label.addEventListener("click", (e) => {
      if (e.target.tagName !== "INPUT") {
        const box = label.querySelector("input");
        box.checked = !box.checked;
      }
      label.style.borderColor = label.querySelector("input").checked ? "" : "";
    });
    list.appendChild(label);
  });
}

$("#btn-generate").addEventListener("click", async () => {
  const ids = [...document.querySelectorAll("#style-list input:checked")].map((b) => b.value);
  if (!ids.length) { showError("至少选择一款发型"); return; }
  const btn = $("#btn-generate");
  btn.disabled = true;
  try {
    const form = new FormData();
    form.append("photo", state.file);
    form.append("style_ids", ids.join(","));
    const resp = await fetch("/v1/generate", { method: "POST", body: form });
    if (!resp.ok) throw new Error(await detail(resp));
    state.jobId = (await resp.json()).job_id;
    renderPending(ids);
    setStep(3);
    pollJob();
  } catch (err) {
    showError(`提交生成失败：${err.message}`);
  } finally {
    btn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// 叁 · 轮询进度 + 图版
// ---------------------------------------------------------------------------

function renderPending(ids) {
  const list = $("#result-list");
  list.innerHTML = "";
  ids.forEach((id, i) => {
    const card = document.createElement("div");
    card.className = "plate";
    card.id = `result-${id}`;
    card.style.animationDelay = `${i * 0.08}s`;
    card.innerHTML = `
      <div class="plate-head">
        <span class="plate-name">${styleName(id)}</span>
        <span class="plate-no">PLATE ${String(i + 1).padStart(2, "0")}</span>
      </div>
      <div class="plate-foot">
        <span class="scores"><span class="dot-spin"></span>渲染中，约 30–60 秒…</span>
      </div>`;
    list.appendChild(card);
  });
}

async function pollJob() {
  clearTimeout(state.pollTimer);
  const started = Date.now();
  const tick = async () => {
    try {
      const resp = await fetch(`/v1/jobs/${state.jobId}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const job = await resp.json();
      if (job.status === "done") { renderResults(job); return; }
      if (job.status === "failed") {
        showError(`生成失败：${job.error ?? "未知错误"}`);
        markPendingErrors();
        return;
      }
      if (Date.now() - started > 5 * 60 * 1000) {
        showError("生成超时，请稍后用「换一批再试」重新提交");
        return;
      }
      state.pollTimer = setTimeout(tick, 2500);
    } catch (err) {
      showError(`查询进度失败：${err.message}`);
    }
  };
  tick();
}

function markPendingErrors() {
  for (const card of document.querySelectorAll("#result-list .plate")) {
    const score = card.querySelector(".scores");
    if (score?.querySelector(".dot-spin")) {
      score.textContent = "未完成";
      score.classList.add("bad");
    }
  }
}

function renderResults(job) {
  state.generated = new Set();
  job.results.forEach((r, i) => {
    const card = $(`#result-${r.style_id}`);
    if (!card) return;
    if (r.error || !r.image_paths.length) {
      card.querySelector(".scores").innerHTML =
        `<span class="bad">本款生成失败：${r.error ?? "无输出"}</span>`;
      return;
    }
    state.generated.add(r.style_id);
    const qc = r.qc_skipped
      ? "未质检，直接交付"
      : `脸型一致 ${r.qc_face_consistency}/10 · 风格还原 ${r.qc_style_match}/10` +
        (r.retry_used ? " · 已修正重试" : "");
    const foot = card.querySelector(".plate-foot");
    foot.innerHTML = `<span class="scores">${qc}</span>`;
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = styleName(r.style_id);
    img.src = `/v1/jobs/${job.id}/results/${r.style_id}`;
    img.addEventListener("click", () => window.open(img.src, "_blank"));
    card.insertBefore(img, foot);
  });
}

$("#btn-report").addEventListener("click", async () => {
  const btn = $("#btn-report");
  btn.disabled = true;
  try {
    // 未成功生成的推荐款 → 报告卡中以参考图 + ✗ 展示（零生成成本）
    const rejected = state.rec.recommended
      .filter((r) => !(state.generated ?? new Set()).has(r.id))
      .map((r) => `${r.id}:${r.reason.replace(/[|:]/g, "，")}`)
      .join("|");
    const form = new FormData();
    form.append("job_id", state.jobId);
    form.append("analysis_json", JSON.stringify(state.rec));
    form.append("rejected", rejected);
    const resp = await fetch("/v1/report", { method: "POST", body: form });
    if (!resp.ok) throw new Error(await detail(resp));
    $("#report-img").src = URL.createObjectURL(await resp.blob());
    $("#report-wrap").hidden = false;
    $("#report-wrap").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    showError(`报告卡生成失败：${err.message}`);
  } finally {
    btn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// 初始化
// ---------------------------------------------------------------------------

loadStyles();
