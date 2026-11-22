import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";
import { Upload, TrendingUp, ShieldCheck, Sliders, AlertTriangle, CheckCircle2 } from "lucide-react";
import "./style.css";

const API_BASE = window.location.origin.replace("-5173.", "-8000.");

function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [file, setFile] = useState(null);
  const [budget, setBudget] = useState(50000);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);
  const [segmentFilter, setSegmentFilter] = useState("ALL");

  const runPipeline = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a CSV file (such as synthetic_campaign_data.csv) first.");
      return;
    }
    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("budget", String(budget));

      const res = await fetch(`${API_BASE}/analyze/full`, {
        method: "POST",
        body: formData,
      });

      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload.detail || "Backend analysis failed.");
      }
      setData(payload);
    } catch (err) {
      setError(err.message || "Failed to reach backend API. Ensure port 8000 is running and Public.");
    } finally {
      setLoading(false);
    }
  };

  const summary = data?.summary;
  const causal = data?.causal;
  const opt = data?.optimization;
  const drift = causal?.drift_report;

  const filteredCustomers = (causal?.customers || []).filter((row) =>
    segmentFilter === "ALL" ? true : row.causal_segment === segmentFilter
  );

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <TrendingUp size={22} />
          <span>EconoCausal</span>
        </div>
        <nav className="nav-list">
          <button
            className={`nav-item ${activeTab === "overview" ? "active" : ""}`}
            onClick={() => setActiveTab("overview")}
          >
            Overview & Data
          </button>
          <button
            className={`nav-item ${activeTab === "causal" ? "active" : ""}`}
            onClick={() => setActiveTab("causal")}
          >
            Causal Analysis (DoWhy & DML)
          </button>
          <button
            className={`nav-item ${activeTab === "optimization" ? "active" : ""}`}
            onClick={() => setActiveTab("optimization")}
          >
            Budget Optimization (MILP)
          </button>
        </nav>
        <div className="sidebar-note">
          <strong>Ref:</strong> ITS/DSML/1186
          <br />
          DML + MILP Dynamic Pricing
        </div>
      </aside>

      <main className="main-content">
        <header className="page-header">
          <div>
            <h1>Dynamic Pricing & Causal Uplift Platform</h1>
            <p className="subtitle">
              Double Machine Learning (EconML), DoWhy Causal Refutation & SciPy MILP Budget Allocation
            </p>
          </div>
        </header>

        <form className="control-card" onSubmit={runPipeline}>
          <div className="control-row">
            <div className="control-group">
              <label htmlFor="csv-upload">
                <Upload size={16} /> Campaign Dataset CSV
              </label>
              <input
                id="csv-upload"
                type="file"
                accept=".csv"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
            </div>

            <div className="control-group">
              <label htmlFor="budget-input">
                <Sliders size={16} /> Promotional Budget Cap ($)
              </label>
              <input
                id="budget-input"
                type="number"
                min="100"
                step="500"
                value={budget}
                onChange={(e) => setBudget(Number(e.target.value) || 0)}
              />
            </div>

            <button type="submit" className="primary-btn" disabled={loading}>
              {loading ? "Running Causal & MILP Pipeline..." : "Run Full Analysis"}
            </button>
          </div>
        </form>

        {error && (
          <div className="alert error-alert">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        )}

        {data && (
          <>
            <div className="banner-grid">
              <div className="banner roi-banner">
                <CheckCircle2 size={18} />
                <span>{opt.roi_summary}</span>
              </div>
              <div className={`banner ${drift.drift_detected ? "warn-banner" : "ok-banner"}`}>
                <ShieldCheck size={18} />
                <span>{drift.summary}</span>
              </div>
            </div>

            <div className="kpi-grid">
              <div className="kpi-card">
                <div className="kpi-label">TOTAL CUSTOMERS</div>
                <div className="kpi-value">{summary.rows.toLocaleString()}</div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">ESTIMATED ATE (DML)</div>
                <div className="kpi-value">{(causal.estimated_ate * 100).toFixed(2)}%</div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">PERSUADABLES TARGETED</div>
                <div className="kpi-value">{opt.selected_customer_count.toLocaleString()}</div>
              </div>
              <div className="kpi-card">
                <div className="kpi-label">EXPECTED NET PROFIT</div>
                <div className="kpi-value">${opt.expected_incremental_profit.toLocaleString()}</div>
              </div>
            </div>

            {activeTab === "overview" && (
              <section className="section-grid">
                <div className="panel">
                  <h3>Dataset & Data Drift Monitor (KS-Test)</h3>
                  <p className="muted">
                    File: <strong>{summary.filename}</strong> · Columns: <strong>{summary.columns}</strong> ·
                    Nulls: <strong>{summary.missing_values}</strong> · Baseline Conversion:{" "}
                    <strong>{summary.purchase_rate_percent}%</strong>
                  </p>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th>KS Stat</th>
                        <th>p-value</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {drift.features.map((f) => (
                        <tr key={f.feature}>
                          <td>{f.feature}</td>
                          <td>{f.ks_stat}</td>
                          <td>{f.p_value}</td>
                          <td>
                            <span className={`badge ${f.status === "Healthy" ? "badge-ok" : "badge-warn"}`}>
                              {f.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="panel">
                  <h3>Causal Quadrant Breakdown</h3>
                  <div className="chart-box">
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={causal.segment_breakdown}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="segment" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#2563eb" radius={[6, 6, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </section>
            )}

            {activeTab === "causal" && (
              <section className="section-stack">
                <div className="section-grid">
                  <div className="panel">
                    <h3>Cumulative Uplift / Qini Curve</h3>
                    <div className="chart-box">
                      <ResponsiveContainer width="100%" height={280}>
                        <LineChart data={causal.qini_curve}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="percentile" unit="%" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="dml_uplift"
                            name="DML Targeted Uplift"
                            stroke="#2563eb"
                            strokeWidth={2.5}
                          />
                          <Line
                            type="monotone"
                            dataKey="random_baseline"
                            name="Random Rollout Baseline"
                            stroke="#94a3b8"
                            strokeDasharray="5 5"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="panel">
                    <h3>DoWhy Causal Refutation Audits</h3>
                    <div className="refute-card">
                      <strong>Random Common Cause</strong>
                      <p>
                        New Effect: <strong>{causal.refutations.random_common_cause.new_effect}</strong> |
                        p-value: <strong>{causal.refutations.random_common_cause.p_value}</strong> | Status:{" "}
                        <span className="badge badge-ok">
                          {causal.refutations.random_common_cause.passed ? "Passed" : "Review"}
                        </span>
                      </p>
                    </div>
                    <div className="refute-card">
                      <strong>Placebo Treatment Refuter</strong>
                      <p>
                        Placebo Effect: <strong>{causal.refutations.placebo_treatment.new_effect}</strong> |
                        p-value: <strong>{causal.refutations.placebo_treatment.p_value}</strong> | Status:{" "}
                        <span className="badge badge-ok">
                          {causal.refutations.placebo_treatment.passed ? "Passed" : "Review"}
                        </span>
                      </p>
                    </div>
                    <div className="dag-box">
                      <strong>Causal DAG Specification:</strong> [{causal.dag.confounders.join(", ")}] →{" "}
                      {causal.dag.treatment} → {causal.dag.outcome}
                    </div>
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-header-row">
                    <h3>Customer Individual Treatment Effect (ITE) Scores</h3>
                    <select
                      value={segmentFilter}
                      onChange={(e) => setSegmentFilter(e.target.value)}
                      className="filter-select"
                    >
                      <option value="ALL">All Causal Segments</option>
                      <option value="Persuadables">Persuadables</option>
                      <option value="Sure Things">Sure Things</option>
                      <option value="Lost Causes">Lost Causes</option>
                      <option value="Sleeping Dogs">Sleeping Dogs</option>
                    </select>
                  </div>
                  <div className="table-scroll">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Customer ID</th>
                          <th>Segment</th>
                          <th>Region</th>
                          <th>List Price ($)</th>
                          <th>Estimated ITE</th>
                          <th>Causal Quadrant</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredCustomers.slice(0, 50).map((c, idx) => (
                          <tr key={`${c.customer_id}-${idx}`}>
                            <td>{c.customer_id}</td>
                            <td>{c.customer_segment}</td>
                            <td>{c.region}</td>
                            <td>${c.list_price}</td>
                            <td>{c.estimated_ite}</td>
                            <td>
                              <span className="badge badge-ok">{c.causal_segment}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </section>
            )}

            {activeTab === "optimization" && (
              <section className="section-stack">
                <div className="kpi-grid">
                  <div className="kpi-card">
                    <div className="kpi-label">BUDGET SPENT</div>
                    <div className="kpi-value">${opt.total_budget_spent.toLocaleString()}</div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-label">BUDGET UTILIZATION</div>
                    <div className="kpi-value">{opt.budget_utilization_percent}%</div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-label">ROI MULTIPLIER</div>
                    <div className="kpi-value">{opt.roi_multiplier}x</div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-label">SAVINGS VS. BLANKET</div>
                    <div className="kpi-value">${opt.savings_vs_blanket.toLocaleString()}</div>
                  </div>
                </div>

                <div className="panel">
                  <h3>SciPy MILP Personalized Discount Allocation Matrix</h3>
                  <div className="table-scroll">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Customer ID</th>
                          <th>Segment</th>
                          <th>Region</th>
                          <th>ITE</th>
                          <th>Discount Cost ($)</th>
                          <th>Expected Net Profit ($)</th>
                          <th>MILP Decision</th>
                        </tr>
                      </thead>
                      <tbody>
                        {opt.allocation_matrix.slice(0, 60).map((r, idx) => (
                          <tr key={`${r.customer_id}-${idx}`}>
                            <td>{r.customer_id}</td>
                            <td>{r.customer_segment}</td>
                            <td>{r.region}</td>
                            <td>{r.estimated_ite}</td>
                            <td>${r.discount_cost}</td>
                            <td>${r.expected_net_profit}</td>
                            <td>
                              <span className={`badge ${r.targeted ? "badge-ok" : "badge-muted"}`}>
                                {r.action}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
