"use strict";

const SUMMARY_URL = "/acceptance/api/summary";
const JUDGE_URL = "/acceptance/api/judge/queries";

const STATUS_TEXT = {
  PASS: "✓ PASS",
  PARTIAL: "◐ PARTIAL",
  FAIL: "✕ FAIL",
  BLOCKED: "⊘ BLOCKED",
  MISSING: "— MISSING",
  "NOT RUN": "· NOT RUN",
  NOT_RUN: "· NOT RUN",
  "NOT APPLICABLE": "— NOT APPLICABLE",
  SUCCESS: "✓ SUCCESS",
  FAILED: "✕ FAILED",
  EMPTY_CONTEXT: "· EMPTY CONTEXT",
  EMPTY_ANSWER: "· EMPTY ANSWER",
  INVALID_JSON: "✕ INVALID JSON",
  INVALID_SCHEMA: "✕ INVALID SCHEMA",
  OUT_OF_RANGE: "✕ OUT OF RANGE",
  UNKNOWN_CONTEXT_ID: "✕ UNKNOWN CONTEXT ID",
  PROVIDER_ERROR: "✕ PROVIDER ERROR",
  RAG_FAILURE: "✕ RAG FAILURE",
  ERROR: "! 不可用",
  LOADING: "… 加载中",
};

const CHECK_STATUS = {
  pass: "PASS",
  fail: "FAIL",
  not_applicable: "NOT APPLICABLE",
};

const state = {
  summary: null,
  judgeQueries: null,
  judgeLoaded: false,
  judgeLoading: false,
};

function el(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (text !== null && text !== undefined) {
    node.textContent = String(text);
  }
  return node;
}

function appendAll(parent, children) {
  for (const child of children) {
    if (child === null || child === undefined) {
      continue;
    }
    if (Array.isArray(child)) {
      appendAll(parent, child);
    } else {
      parent.append(child);
    }
  }
  return parent;
}

function statusKey(status) {
  if (status === null || status === undefined || status === "") {
    return "MISSING";
  }
  return String(status).toUpperCase();
}

function statusClass(status) {
  switch (statusKey(status)) {
    case "PASS":
      return "is-pass";
    case "PARTIAL":
      return "is-partial";
    case "FAIL":
      return "is-fail";
    case "BLOCKED":
      return "is-blocked";
    case "NOT RUN":
    case "NOT_RUN":
      return "is-not-run";
    case "SUCCESS":
      return "is-pass";
    case "FAILED":
    case "INVALID_JSON":
    case "INVALID_SCHEMA":
    case "OUT_OF_RANGE":
    case "UNKNOWN_CONTEXT_ID":
    case "PROVIDER_ERROR":
    case "RAG_FAILURE":
      return "is-fail";
    case "EMPTY_CONTEXT":
    case "EMPTY_ANSWER":
      return "is-warning";
    case "NOT APPLICABLE":
      return "is-not-applicable";
    case "ERROR":
      return "is-error";
    case "LOADING":
      return "is-loading";
    case "MISSING":
      return "is-missing";
    default:
      return "is-neutral";
  }
}

function badge(status, large = false) {
  const key = statusKey(status);
  const node = el("span", `badge ${statusClass(key)}`, STATUS_TEXT[key] || `· ${key}`);
  if (large) {
    node.classList.add("badge-lg");
  }
  return node;
}

function formatValue(value, digits = 4) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return "—";
    }
    return value.toFixed(digits);
  }
  return String(value);
}

function formatCount(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  return String(value);
}

function formatMs(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return `${value.toFixed(2)} ms`;
  }
  return String(value);
}

function formatBool(value) {
  if (value === true) {
    return "是";
  }
  if (value === false) {
    return "否";
  }
  return "—";
}

function formatDate(value) {
  return value ? String(value) : "—";
}

function paramsText(entries) {
  const parts = entries
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([label, value]) => `${label} = ${value}`);
  return parts.length ? parts.join(" · ") : "参数未记录";
}

function sectionCard(title, meta, children, className = "") {
  const card = el("article", `panel-card ${className}`.trim());
  const heading = el("div", "section-heading");
  heading.append(el("h2", "", title));
  if (meta instanceof Node) {
    heading.append(meta);
  } else if (meta) {
    heading.append(el("span", "section-meta", meta));
  }
  card.append(heading);
  appendAll(card, children);
  return card;
}

function missingSection(label, section) {
  const reason = section && section.reason
    ? section.reason
    : "artifact 缺失，页面不显示数值。";
  return sectionCard(
    label,
    badge("MISSING"),
    [el("p", "", reason)],
    "warning-panel"
  );
}

function table(caption, columns, rows) {
  const wrap = el("div", "table-wrap");
  const node = el("table");
  node.append(el("caption", "", caption));
  const thead = el("thead");
  const headRow = el("tr");
  for (const column of columns) {
    const th = el("th", column.className || "", column.label);
    th.scope = "col";
    headRow.append(th);
  }
  thead.append(headRow);
  node.append(thead);

  const tbody = el("tbody");
  if (!rows.length) {
    const row = el("tr");
    const cell = el("td", "muted", "没有可显示的记录。");
    cell.colSpan = columns.length;
    row.append(cell);
    tbody.append(row);
  } else {
    for (const record of rows) {
      const row = el("tr");
      for (const column of columns) {
        const cell = el("td", column.className || "");
        const value = column.value(record);
        if (value instanceof Node) {
          cell.append(value);
        } else if (value === null || value === undefined || value === "") {
          cell.textContent = "—";
        } else {
          cell.textContent = String(value);
        }
        row.append(cell);
      }
      tbody.append(row);
    }
  }
  node.append(tbody);
  wrap.append(node);
  return wrap;
}

function definitionList(entries) {
  const list = el("dl", "definition-list");
  for (const [label, value] of entries) {
    list.append(el("dt", "", label));
    const detail = el("dd");
    if (value instanceof Node) {
      detail.append(value);
    } else {
      detail.textContent = value === null || value === undefined || value === ""
        ? "—"
        : String(value);
    }
    list.append(detail);
  }
  return list;
}

function codeBlock(values) {
  const block = el("div", "stack");
  for (const [label, value] of values) {
    const line = el("div");
    line.append(el("span", "muted", `${label}: `));
    line.append(el("code", "", value || "—"));
    block.append(line);
  }
  return block;
}

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function setOverallStatus(status) {
  const current = document.getElementById("overall-status");
  const next = badge(status, true);
  next.id = "overall-status";
  if (current) {
    current.replaceWith(next);
  }
}

function showError() {
  const banner = document.getElementById("error-banner");
  const text = document.getElementById("error-text");
  if (text) {
    text.textContent =
      "无法读取验收数据。请确认 artifact 存在且为合法 JSON，然后重试。";
  }
  if (banner) {
    banner.hidden = false;
  }
  setOverallStatus("ERROR");
  const targets = [
    "overview-meta",
    "checks",
    "artifact-table",
    "dataset",
    "retrieval-content",
    "index-content",
    "chunking-content",
    "rerank-content",
    "judge-content",
    "evidence-content",
  ];
  for (const id of targets) {
    const target = document.getElementById(id);
    if (!target) {
      continue;
    }
    target.replaceChildren(
      el("p", "missing-state", "数据不可用，未显示任何数值。")
    );
  }
  const cards = document.getElementById("summary-cards");
  if (cards) {
    cards.replaceChildren();
  }
}

function hideError() {
  const banner = document.getElementById("error-banner");
  if (banner) {
    banner.hidden = true;
  }
}

async function loadSummary() {
  setOverallStatus("LOADING");
  const meta = document.getElementById("overview-meta");
  if (meta) {
    meta.textContent = "正在读取 artifact…";
  }
  try {
    const response = await fetch(SUMMARY_URL, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`summary HTTP ${response.status}`);
    }
    const data = await response.json();
    if (!data || data.read_only !== true) {
      throw new Error("summary payload is not read-only");
    }
    state.summary = data;
    state.judgeQueries = null;
    state.judgeLoaded = false;
    hideError();
    renderAll(data);
  } catch (error) {
    state.summary = null;
    showError();
  }
}

function renderAll(summary) {
  setOverallStatus(summary.status);
  renderOverview(summary);
  renderRetrieval(summary.sections.retrieval);
  renderIndex(summary.sections.index);
  renderChunking(summary.sections.chunking);
  renderRerank(summary.sections.rerank);
  renderJudge(summary.sections.judge);
  renderEvidence(summary);
  if (location.hash === "#judge") {
    ensureJudgeQueries();
  }
}

function renderOverview(summary) {
  const meta = document.getElementById("overview-meta");
  if (meta) {
    meta.textContent =
      `总状态由 report_status() 从 artifact 字段推导；页面生成于 ${formatDate(summary.generated_at)}。` +
      " 未提供的值显示为 —，不按 0 分处理。";
  }

  const artifacts = safeArray(summary.artifacts);
  const judge = summary.sections.judge || {};
  const judgeSummary = judge.summary || {};
  const dataset = summary.dataset || {};
  const cards = document.getElementById("summary-cards");
  if (cards) {
    const present = artifacts.filter((item) => item.present).length;
    const coverage = judgeSummary.queries !== null &&
      judgeSummary.queries !== undefined &&
      judgeSummary.expected_queries !== null &&
      judgeSummary.expected_queries !== undefined
      ? `${judgeSummary.queries}/${judgeSummary.expected_queries}`
      : "—";
    const notRun = judgeSummary.not_run_queries;
    const realJudge = judge.available ? formatBool(judge.provider && judge.provider.is_real) : "—";

    cards.replaceChildren(
      statCard("总体状态", badge(summary.status, true), summary.status_reason || "—"),
      statCard(
        "Artifact",
        `${present}/${artifacts.length || 0}`,
        `${safeArray(summary.required_artifacts).length} 必需 · ${safeArray(summary.optional_artifacts).length} 可选`
      ),
      statCard(
        "查询数",
        formatCount(dataset.queries),
        `gold 文档 ${formatCount(dataset.gold_docs)}`
      ),
      statCard(
        "Judge 覆盖",
        coverage,
        `NOT RUN ${formatCount(notRun)}`
      ),
      statCard(
        "真实 Judge",
        realJudge,
        judge.available ? `${judge.provider?.kind || "—"} · same_model ${formatBool(judgeSummary.same_model)}` : "judge artifact 缺失"
      )
    );
  }

  const checks = document.getElementById("checks");
  if (checks) {
    const list = el("ul", "check-list");
    for (const check of safeArray(summary.checks)) {
      const item = el("li", "check-item");
      const copy = el("div", "check-copy");
      copy.append(el("span", "check-label", check.label || check.id || "检查"));
      copy.append(el("span", "check-detail", check.detail || "—"));
      item.append(copy, badge(CHECK_STATUS[check.state] || "NOT APPLICABLE"));
      list.append(item);
    }
    checks.replaceChildren(
      sectionCard(
        "自动检查",
        "机械规则，不是人工验收结论",
        [list]
      )
    );
  }

  const artifactTarget = document.getElementById("artifact-table");
  if (artifactTarget) {
    artifactTarget.replaceChildren(
      sectionCard(
        "Artifact 登记",
        `${artifacts.length} 个配置输入`,
        [renderArtifactTable(artifacts)]
      )
    );
  }

  const datasetTarget = document.getElementById("dataset");
  if (datasetTarget) {
    const datasetBlock = el("div");
    datasetBlock.append(el("h3", "", "数据"));
    datasetBlock.append(
      definitionList([
        ["查询数", formatCount(dataset.queries)],
        ["磁盘查询数", formatCount(dataset.eval_queries_on_disk)],
        ["gold 文档数", formatCount(dataset.gold_docs)],
      ])
    );
    const hashes = dataset.hashes || {};
    if (Object.keys(hashes).length) {
      datasetBlock.append(
        codeBlock(Object.entries(hashes).map(([key, value]) => [key, value]))
      );
    }
    const environment = summary.environment || {};
    const environmentBlock = el("div");
    environmentBlock.append(el("h3", "", "环境"));
    environmentBlock.append(
      definitionList([
        ["Python", environment.python_version],
        ["PostgreSQL", environment.postgres_version],
        ["pgvector", environment.vector_version],
        ["Embedding", environment.embedding_model],
        ["Reranker", environment.reranker_model],
        ["文档 / chunks", `${formatCount(environment.documents)} / ${formatCount(environment.chunks)}`],
        ["执行环境", environment.execution_environment],
      ])
    );
    const columns = el("div", "two-column");
    columns.append(datasetBlock, environmentBlock);
    datasetTarget.replaceChildren(
      sectionCard("数据与环境", "来自 artifact 元数据", [columns])
    );
  }
}

function statCard(label, value, note) {
  const card = el("div", "stat-card");
  card.append(el("span", "stat-label", label));
  const valueNode = el("span", "stat-value");
  if (value instanceof Node) {
    valueNode.append(value);
  } else {
    valueNode.textContent = value === null || value === undefined ? "—" : String(value);
  }
  card.append(valueNode);
  if (note) {
    card.append(el("span", "stat-note", note));
  }
  return card;
}

function renderArtifactTable(artifacts) {
  const columns = [
    {
      label: "Artifact",
      value: (row) => {
        const cell = el("div");
        cell.append(el("strong", "", row.key || "—"));
        cell.append(el("span", "muted", ` · ${row.label || "—"}`));
        if (row.required) {
          cell.append(el("span", "tag", "必需"));
        } else {
          cell.append(el("span", "tag", "可选"));
        }
        return cell;
      },
    },
    {
      label: "状态",
      value: (row) => (row.present ? badge(row.status) : badge("MISSING")),
    },
    {
      label: "生成时间",
      className: "mono",
      value: (row) => formatDate(row.generated_at),
    },
    {
      label: "SHA256",
      className: "mono",
      value: (row) => (row.sha256 ? el("code", "", row.sha256) : "—"),
    },
    {
      label: "路径",
      className: "mono",
      value: (row) => (row.path ? el("code", "", row.path) : "—"),
    },
  ];
  return table("Artifact 路径相对仓库根；SHA256 是页面读取的同一份文件。", columns, artifacts);
}

function renderRetrieval(section) {
  const target = document.getElementById("retrieval-content");
  if (!target) {
    return;
  }
  if (!section || !section.available) {
    target.replaceChildren(missingSection("统一检索", section));
    return;
  }
  const params = section.parameters || {};
  const columns = [
    { label: "Arm", value: (row) => row.name || "—" },
    { label: "状态", value: (row) => badge(row.status) },
    { label: "Recall macro", className: "num", value: (row) => formatValue(row.recall_macro) },
    { label: "Recall micro", className: "num", value: (row) => formatValue(row.recall_micro) },
    { label: "P@K macro", className: "num", value: (row) => formatValue(row.precision_macro) },
    { label: "P@K ceiling", className: "num", value: (row) => formatValue(row.precision_ceiling) },
    { label: "nDCG@K", className: "num", value: (row) => formatValue(row.ndcg_macro) },
    { label: "MRR", className: "num", value: (row) => formatValue(row.mrr) },
    { label: "Hit@1", className: "num", value: (row) => formatCount(row.hit_at_1) },
    { label: "Evaluated", className: "num", value: (row) => formatCount(row.evaluated) },
    { label: "Failed", className: "num", value: (row) => formatCount(row.failed) },
    { label: "P50", className: "num", value: (row) => formatMs(row.p50_ms) },
    { label: "P95", className: "num", value: (row) => formatMs(row.p95_ms) },
  ];
  target.replaceChildren(
    sectionCard(
      "统一检索",
      "不重算，字段直接来自 benchmark artifact",
      [
        el("p", "", paramsText([
          ["top_k", params.top_k],
          ["candidate_k", params.candidate_k],
          ["rrf_k", params.rrf_k],
          ["chunk_strategy", params.chunk_strategy],
          ["queries", params.queries],
          ["expected_queries", params.expected_queries],
        ])),
        table("指标为文档级；未显示的值为 —。", columns, safeArray(section.arms)),
      ]
    )
  );
}

function renderIndex(section) {
  const target = document.getElementById("index-content");
  if (!target) {
    return;
  }
  if (!section || !section.available) {
    target.replaceChildren(missingSection("索引参数", section));
    return;
  }
  const configurations = safeArray(section.configurations);
  const settings = [];
  for (const configuration of configurations) {
    for (const key of Object.keys(configuration.query || {})) {
      if (!settings.includes(key)) {
        settings.push(key);
      }
    }
  }
  const columns = [
    { label: "索引", className: "mono", value: (row) => row.name || "—" },
    { label: "方法", value: (row) => row.method || "—" },
    {
      label: "参数",
      className: "mono",
      value: (row) => {
        const entries = Object.entries(row.params || {});
        return entries.length
          ? entries.map(([key, value]) => `${key}=${value}`).join(", ")
          : "—";
      },
    },
    { label: "状态", value: (row) => badge(row.status) },
    { label: "构建", className: "num", value: (row) => formatMs(row.build_ms) },
    { label: "大小", className: "num", value: (row) => row.size_pretty || "—" },
    { label: "行数", className: "num", value: (row) => formatCount(row.row_count) },
  ];
  for (const setting of settings) {
    columns.push({
      label: setting,
      className: "num",
      value: (row) => {
        const measurement = (row.query || {})[setting];
        if (!measurement) {
          return "—";
        }
        return `${formatValue(measurement.recall, 3)} / ${formatMs(measurement.p50_ms)}`;
      },
    });
  }
  const children = [
    el("p", "", paramsText([
      ["top_k", (section.parameters || {}).top_k],
      ["queries", (section.parameters || {}).queries],
      ["chunk_strategy", (section.parameters || {}).chunk_strategy],
      ["repetitions", (section.parameters || {}).repetitions],
    ])),
  ];
  if (section.synthetic) {
    children.push(
      el(
        "p",
        "missing-state",
        `SYNTHETIC × ${formatCount(section.synthetic_scale)}：向量为确定性复制，只反映索引行为，不进入语义质量结论。`
      )
    );
  }
  if (section.baseline) {
    children.push(
      el(
        "p",
        "",
        `精确基线：${section.baseline.method || "—"}，Top-${formatCount(section.baseline.top_k)}，` +
        `${formatCount(section.baseline.queries)} 条查询。`
      )
    );
  }
  children.push(
    table(
      "每个配置在临时表副本上重建索引；ANN Recall 相对精确 Seq Scan。单元格为 ANN Recall / P50。",
      columns,
      configurations
    )
  );
  target.replaceChildren(
    sectionCard("索引参数实验", "ANN Recall 不是语义 Recall", children)
  );
}

function renderChunking(section) {
  const target = document.getElementById("chunking-content");
  if (!target) {
    return;
  }
  if (!section || !section.available) {
    target.replaceChildren(missingSection("分块策略", section));
    return;
  }
  const params = section.parameters || {};
  const columns = [
    { label: "Strategy", value: (row) => row.strategy || "—" },
    { label: "Arm", value: (row) => row.arm || "—" },
    { label: "状态", value: (row) => badge(row.status) },
    { label: "Recall macro", className: "num", value: (row) => formatValue(row.recall_macro) },
    { label: "Recall micro", className: "num", value: (row) => formatValue(row.recall_micro) },
    { label: "P@K macro", className: "num", value: (row) => formatValue(row.precision_macro) },
    { label: "P@K ceiling", className: "num", value: (row) => formatValue(row.precision_ceiling) },
    { label: "nDCG@K", className: "num", value: (row) => formatValue(row.ndcg_macro) },
    { label: "MRR", className: "num", value: (row) => formatValue(row.mrr) },
    { label: "Hit@1", className: "num", value: (row) => formatCount(row.hit_at_1) },
    { label: "Evaluated", className: "num", value: (row) => formatCount(row.evaluated) },
    { label: "P50", className: "num", value: (row) => formatMs(row.p50_ms) },
    { label: "P95", className: "num", value: (row) => formatMs(row.p95_ms) },
  ];
  const stats = Object.entries(section.chunk_stats || {});
  const statsColumns = [
    { label: "Strategy", value: (row) => row[0] },
    { label: "Chunks", className: "num", value: (row) => formatCount(row[1].chunks) },
    { label: "Documents", className: "num", value: (row) => formatCount(row[1].documents) },
    { label: "Avg chars", className: "num", value: (row) => formatValue(row[1].avg_chars, 1) },
    { label: "Min chars", className: "num", value: (row) => formatCount(row[1].min_chars) },
    { label: "Max chars", className: "num", value: (row) => formatCount(row[1].max_chars) },
  ];
  const children = [
    el("p", "", paramsText([
      ["top_k", params.top_k],
      ["candidate_k", params.candidate_k],
      ["rrf_k", params.rrf_k],
      ["queries", params.queries],
    ])),
    table("3 种策略 × 各 arm；表中没有单一策略在所有指标上胜出的结论。", columns, safeArray(section.rows)),
  ];
  if (stats.length) {
    children.push(table("分块规模", statsColumns, stats));
  }
  target.replaceChildren(
    sectionCard("分块策略对比", "所有策略共用同一个 HNSW 索引", children)
  );
}

function renderRerank(section) {
  const target = document.getElementById("rerank-content");
  if (!target) {
    return;
  }
  if (!section || !section.available) {
    target.replaceChildren(missingSection("精排 20→5", section));
    return;
  }
  const params = section.parameters || {};
  const columns = [
    { label: "Arm", value: (row) => row.name || "—" },
    { label: "状态", value: (row) => badge(row.status) },
    { label: "Recall macro", className: "num", value: (row) => formatValue(row.recall_macro) },
    { label: "Recall micro", className: "num", value: (row) => formatValue(row.recall_micro) },
    { label: "P@K macro", className: "num", value: (row) => formatValue(row.precision_macro) },
    { label: "P@K ceiling", className: "num", value: (row) => formatValue(row.precision_ceiling) },
    { label: "nDCG@K", className: "num", value: (row) => formatValue(row.ndcg_macro) },
    { label: "MRR", className: "num", value: (row) => formatValue(row.mrr) },
    { label: "Hit@1", className: "num", value: (row) => formatCount(row.hit_at_1) },
    { label: "Evaluated", className: "num", value: (row) => formatCount(row.evaluated) },
    { label: "P50", className: "num", value: (row) => formatMs(row.p50_ms) },
    { label: "P95", className: "num", value: (row) => formatMs(row.p95_ms) },
  ];
  target.replaceChildren(
    sectionCard(
      "Cross-Encoder 精排",
      "实时检索 + 实时重排",
      [
        el("p", "", paramsText([
          ["候选池", params.candidate_k],
          ["最终 Top-K", params.top_k],
          ["rrf_k", params.rrf_k],
          ["chunk_strategy", params.chunk_strategy],
          ["queries", params.queries],
        ])),
        table(
          "两个臂共享同一候选池；重排只改变排序，候选池决定 Recall。",
          columns,
          safeArray(section.arms)
        ),
        el("p", "", "延迟包含 CPU Cross-Encoder 前向，单进程、顺序执行，不代表并发吞吐。"),
      ]
    )
  );
}

function renderJudge(section) {
  const target = document.getElementById("judge-content");
  if (!target) {
    return;
  }
  if (!section || !section.available) {
    target.replaceChildren(missingSection("LLM-as-a-Judge", section));
    return;
  }
  const summary = section.summary || {};
  const provider = section.provider || {};
  const rag = section.rag || {};
  const metrics = section.metrics || {};
  const children = [];

  const banner = el("div", "panel-card warning-panel");
  banner.append(
    el("h2", "", "LLM-as-a-Judge"),
    el(
      "p",
      "",
      `只评判 ${formatCount(summary.queries)} / ${formatCount(summary.expected_queries)} 条查询；` +
      `${formatCount(summary.not_run_queries)} 条 NOT RUN，未评分且未按 0 分计入。`
    )
  );
  const tags = el("div", "tag-row");
  tags.append(
    el("span", "tag", `judge ${provider.kind || "—"}`),
    el("span", "tag", `model ${provider.model || "—"}`),
    el("span", "tag", `rag ${rag.kind || "—"} / ${rag.retrieval_mode || "—"}`),
    el("span", "tag", `real calls ${formatBool(provider.real_judge_calls)}`),
    el("span", "tag", `same model ${formatBool(summary.same_model)}`)
  );
  banner.append(tags);
  if (summary.same_model) {
    banner.append(
      el(
        "p",
        "",
        "同模型自评：偏差未量化，分数不能视为独立判断，也不是人工真值。"
      )
    );
  }
  if (provider.is_real !== true) {
    banner.append(
      el("p", "", "本次不是真实模型调用；不能把该结果描述为真实 LLM 评估。")
    );
  }
  children.push(banner);

  const metricRows = [
    ["Context Relevance", metrics.context_relevance || {}],
    ["Faithfulness", metrics.faithfulness || {}],
  ];
  const metricColumns = [
    { label: "Metric", value: (row) => row[0] },
    { label: "Attempted", className: "num", value: (row) => formatCount(row[1].attempted) },
    { label: "Success", className: "num", value: (row) => formatCount(row[1].success) },
    { label: "Failed", className: "num", value: (row) => formatCount(row[1].failed) },
    { label: "Success rate", className: "num", value: (row) => formatValue(row[1].success_rate) },
    { label: "Valid n", className: "num", value: (row) => formatCount(row[1].valid_scores) },
    { label: "Mean", className: "num", value: (row) => formatValue(row[1].mean) },
    { label: "Median", className: "num", value: (row) => formatValue(row[1].median) },
    { label: "Low count", className: "num", value: (row) => formatCount(row[1].low_score_count) },
  ];
  children.push(
    sectionCard(
      "指标汇总",
      "均值/中位数/分布只使用成功评分",
      [table("失败与未运行不计 0 分。", metricColumns, metricRows)]
    )
  );

  const distribution = el("div", "dist-grid");
  for (const [label, item] of metricRows) {
    const card = el("div", "dist-card");
    card.append(el("h3", "", label));
    const counts = item.distribution || {};
    const maxCount = Math.max(
      1,
      ...["1", "2", "3", "4", "5"].map((score) => Number(counts[score] || 0))
    );
    for (const score of ["1", "2", "3", "4", "5"]) {
      const count = Number(counts[score] || 0);
      const row = el("div", "dist-row");
      row.append(el("span", "dist-score", score));
      const bar = el("progress", "dist-bar");
      bar.max = maxCount;
      bar.value = count;
      bar.setAttribute("aria-label", `${label} 分数 ${score} 的数量：${count}`);
      row.append(bar, el("span", "dist-count", count));
      card.append(row);
    }
    distribution.append(card);
  }
  children.push(
    sectionCard("分数分布", "横条按该指标内的最大计数缩放", [distribution])
  );

  const lowRows = [];
  let lowCount = 0;
  for (const [label, item] of metricRows) {
    if (typeof item.low_score_count === "number") {
      lowCount += item.low_score_count;
    }
    for (const sample of safeArray(item.low_score_samples)) {
      lowRows.push([label, sample]);
    }
  }
  if (lowRows.length) {
    const lowColumns = [
      { label: "Metric", value: (row) => row[0] },
      { label: "Query", value: (row) => row[1].query_id || "—" },
      { label: "Score", className: "num", value: (row) => formatCount(row[1].score) },
      { label: "Reason", value: (row) => row[1].reason || "—" },
    ];
    children.push(
      sectionCard("低分样例", `${lowRows.length} 条样例`, [
        table("只列出 artifact 已记录的低分样例。", lowColumns, lowRows),
      ])
    );
  } else if (lowCount > 0) {
    children.push(
      sectionCard("低分样例", `${lowCount} 条`, [
        el("p", "missing-state", "artifact 记录了低分，但没有逐条样例；不以空表代替。"),
      ])
    );
  } else {
    children.push(
      sectionCard("低分样例", "0 条", [
        el("p", "", "没有低于阈值的成功评分。"),
      ])
    );
  }

  const failures = [];
  for (const [label, item] of metricRows) {
    const counts = item.failure_counts || {};
    for (const [status, count] of Object.entries(counts.status || {})) {
      failures.push([label, status, count]);
    }
    for (const [category, count] of Object.entries(counts.error_category || {})) {
      failures.push([label, `provider:${category}`, count]);
    }
  }
  const failureColumns = [
    { label: "Metric", value: (row) => row[0] },
    { label: "状态 / 错误", value: (row) => row[1] },
    { label: "数量", className: "num", value: (row) => formatCount(row[2]) },
  ];
  children.push(
    sectionCard("失败分类", failures.length ? `${failures.length} 类` : "0 类", [
      table(
        "失败与未运行不进入均值，也不折算成 0 分。",
        failureColumns,
        failures
      ),
    ])
  );

  const limitations = el("ul", "list-plain");
  for (const item of safeArray(section.limitations)) {
    limitations.append(el("li", "", item));
  }
  if (!limitations.childElementCount) {
    limitations.append(el("li", "", "该 artifact 未记录限制。"));
  }
  children.push(sectionCard("Judge 限制", "artifact 原文", [limitations]));

  const queryCard = el("article", "panel-card");
  queryCard.append(
    el("h2", "", "逐查询结果"),
    el(
      "p",
      "",
      `共 ${formatCount(summary.queries)} 条已评分记录；逐条理由与 claim 通过接口按需读取，` +
      "原始 Provider 响应不会返回。"
    )
  );
  const button = el("button", "button", "加载逐查询结果");
  button.type = "button";
  button.id = "load-judge-button";
  button.addEventListener("click", () => loadJudgeQueries());
  const results = el("div", "stack");
  results.id = "judge-query-results";
  queryCard.append(button, results);
  children.push(queryCard);

  target.replaceChildren(...children);
  if (state.judgeQueries) {
    renderJudgeQueries(state.judgeQueries);
  }
}

async function loadJudgeQueries() {
  if (state.judgeLoading) {
    return;
  }
  const button = document.getElementById("load-judge-button");
  const results = document.getElementById("judge-query-results");
  if (button) {
    button.disabled = true;
    button.textContent = "正在读取…";
  }
  state.judgeLoading = true;
  try {
    const response = await fetch(JUDGE_URL, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`judge HTTP ${response.status}`);
    }
    const data = await response.json();
    state.judgeQueries = data;
    state.judgeLoaded = true;
    renderJudgeQueries(data);
  } catch (error) {
    if (results) {
      results.replaceChildren(
        el(
          "p",
          "missing-state",
          "逐查询结果读取失败。可以重试；页面不会用其他数据替代。"
        )
      );
    }
  } finally {
    state.judgeLoading = false;
    if (button) {
      button.disabled = false;
      button.textContent = "重新读取逐查询结果";
    }
  }
}

function ensureJudgeQueries() {
  if (!state.summary || !state.summary.sections.judge.available) {
    return;
  }
  if (state.judgeLoaded || state.judgeLoading) {
    return;
  }
  loadJudgeQueries();
}

function renderJudgeQueries(data) {
  const results = document.getElementById("judge-query-results");
  if (!results) {
    return;
  }
  if (!data || !data.available) {
    results.replaceChildren(
      el("p", "missing-state", "judge artifact 缺失，没有逐查询结果可显示。")
    );
    return;
  }
  const records = safeArray(data.records);
  const intro = el(
    "p",
    "",
    `返回 ${records.length} 条，共 ${formatCount(data.total)} 条；` +
    `预期 ${formatCount(data.expected)} 条，NOT RUN ${formatCount(data.not_run)} 条。`
  );
  const list = el("div");
  for (const record of records) {
    list.append(renderJudgeRecord(record));
  }
  if (!records.length) {
    list.append(el("p", "missing-state", "没有可显示的逐查询记录。"));
  }
  results.replaceChildren(intro, list);
}

function renderJudgeRecord(record) {
  const details = el("details");
  const summary = el("summary");
  const first = record.context_relevance || {};
  const second = record.faithfulness || {};
  summary.append(
    el("strong", "", record.query_id || "—"),
    el("span", "muted", ` · ${record.query || "—"}`)
  );
  const tags = el("div", "tag-row");
  tags.append(
    el("span", "tag", `RAG ${record.rag_status || "—"}`),
    el("span", "tag", `CR ${formatCount(first.score)}`),
    el("span", "tag", `Faithfulness ${formatCount(second.score)}`),
    el("span", "tag", `unsupported ${safeArray(second.unsupported_claims).length}`)
  );
  summary.append(tags);
  details.append(summary);

  const body = el("div", "details-body");
  body.append(judgeMetricBlock("Context Relevance", first, false));
  body.append(judgeMetricBlock("Faithfulness", second, true));
  details.append(body);
  return details;
}

function judgeMetricBlock(label, item, withClaims) {
  const card = el("div");
  const heading = el("div", "section-heading");
  heading.append(el("h3", "", label));
  heading.append(badge(item.status || "NOT RUN"));
  card.append(heading);
  const meta = el("p", "", paramsText([
    ["score", formatCount(item.score)],
    ["model", item.model],
    ["latency", formatMs(item.latency_ms)],
    ["attempts", formatCount(item.attempts)],
    ["prompt", item.prompt_version],
  ]));
  card.append(meta);
  if (item.reason) {
    card.append(el("p", "", item.reason));
  }
  if (safeArray(item.relevant_context_ids).length) {
    const tags = el("div", "tag-row");
    for (const id of item.relevant_context_ids) {
      tags.append(el("span", "tag", id));
    }
    card.append(el("p", "", "相关上下文："), tags);
  }
  if (safeArray(item.irrelevant_context_ids).length) {
    const tags = el("div", "tag-row");
    for (const id of item.irrelevant_context_ids) {
      tags.append(el("span", "tag", id));
    }
    card.append(el("p", "", "判为不相关："), tags);
  }
  if (withClaims) {
    const claims = safeArray(item.claims);
    if (claims.length) {
      const list = el("ul", "claim-list");
      for (const claim of claims) {
        list.append(
          el(
            "li",
            "",
            `${claim.support_status || "—"}：${claim.claim || "—"}` +
            (safeArray(claim.evidence_context_ids).length
              ? `（${claim.evidence_context_ids.join(", ")}）`
              : "")
          )
        );
      }
      card.append(el("p", "", "Claims："), list);
    }
    if (safeArray(item.unsupported_claims).length) {
      const list = el("ul", "claim-list");
      for (const claim of item.unsupported_claims) {
        list.append(el("li", "", claim));
      }
      card.append(el("p", "", "Unsupported claims："), list);
    }
  }
  if (item.error) {
    card.append(
      el(
        "p",
        "muted",
        `错误分类：${item.error.category || "—"} · ` +
        `status ${formatCount(item.error.status_code)} · ` +
        `retryable ${formatBool(item.error.retryable)}`
      )
    );
  }
  return card;
}

function renderEvidence(summary) {
  const target = document.getElementById("evidence-content");
  if (!target) {
    return;
  }
  const children = [];
  children.push(
    sectionCard(
      "Artifact 证据",
      "完整 SHA256 与仓库相对路径",
      [renderArtifactTable(safeArray(summary.artifacts))]
    )
  );

  const limitations = safeArray(summary.limitations);
  if (limitations.length) {
    const columns = [
      { label: "来源", value: (row) => row.source || "—" },
      { label: "限制", value: (row) => row.text || "—" },
    ];
    children.push(
      sectionCard("artifact 声明的限制", `${limitations.length} 条`, [
        table("按 artifact 原样展示，不做改写。", columns, limitations),
      ])
    );
  } else {
    children.push(
      sectionCard("artifact 声明的限制", "0 条", [
        el("p", "", "已加载 artifact 没有记录 limitations。"),
      ])
    );
  }

  const caveats = el("ul", "list-plain");
  for (const item of safeArray(summary.caveats)) {
    caveats.append(el("li", "", item));
  }
  children.push(sectionCard("页面声明", "固定文案", [caveats]));

  children.push(
    sectionCard(
      "未验证范围",
      "不要从页面推断生产结论",
      [
        el(
          "p",
          "",
          "真实 LLM 全量 30 条、Answer Relevance、人工 Judge 校准、并发与容量、" +
          "10 万级以上索引调优均未由本页验证。"
        ),
        el(
          "p",
          "",
          "浏览器与容器验证状态以 README / roadmap 的实际记录为准；页面本身不宣称通过。"
        ),
        el(
          "p",
          "",
          "本页只读，自动检查是机械规则，不是人工验收结论。"
        ),
      ]
    )
  );
  target.replaceChildren(...children);
}

function activatePanel(panelName, updateHash = true) {
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
  const targetTab = document.getElementById(`tab-${panelName}`);
  if (!targetTab) {
    return;
  }
  for (const tab of tabs) {
    const selected = tab === targetTab;
    tab.setAttribute("aria-selected", selected ? "true" : "false");
    tab.tabIndex = selected ? 0 : -1;
    const panel = document.getElementById(tab.getAttribute("aria-controls"));
    if (panel) {
      panel.hidden = !selected;
    }
  }
  if (updateHash) {
    history.replaceState(null, "", `#${panelName}`);
  }
  if (panelName === "judge") {
    ensureJudgeQueries();
  }
}

function setupTabs() {
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
  for (const tab of tabs) {
    tab.addEventListener("click", () => {
      activatePanel(tab.dataset.panel);
    });
    tab.addEventListener("keydown", (event) => {
      const index = tabs.indexOf(tab);
      let next = null;
      if (event.key === "ArrowRight") {
        next = tabs[(index + 1) % tabs.length];
      } else if (event.key === "ArrowLeft") {
        next = tabs[(index - 1 + tabs.length) % tabs.length];
      } else if (event.key === "Home") {
        next = tabs[0];
      } else if (event.key === "End") {
        next = tabs[tabs.length - 1];
      }
      if (next) {
        event.preventDefault();
        activatePanel(next.dataset.panel);
        next.focus();
      }
    });
  }
  window.addEventListener("hashchange", () => {
    const panel = location.hash.replace("#", "") || "overview";
    activatePanel(panel, false);
  });
  const initial = location.hash.replace("#", "") || "overview";
  activatePanel(initial, false);
}

function setupRetry() {
  const button = document.getElementById("retry-button");
  if (button) {
    button.addEventListener("click", () => loadSummary());
  }
}

function init() {
  setupTabs();
  setupRetry();
  loadSummary();
}

init();
