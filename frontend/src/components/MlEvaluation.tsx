import { useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import {
  Activity,
  CheckCircle2,
  Gauge,
  GitCompareArrows,
  Layers,
  LineChart as LineChartIcon,
  XCircle,
} from "lucide-react";
import StatCard from "./StatCard";

const AXIS = { stroke: "#64748b", fontSize: 11 };
const GRID = "rgba(255,255,255,0.06)";
const TOOLTIP = {
  contentStyle: {
    background: "var(--bg-surface)",
    border: "1px solid rgba(59,130,246,0.3)",
    borderRadius: 8,
    fontSize: 12,
  },
  labelStyle: { color: "var(--text-main)" },
} as const;

const VERDICT_OK = "ACEPTADA";
const VERDICT_NO = "RECHAZADA";

const pct = (v: number | null | undefined, d = 2) =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(d)} %`;
const num = (v: number | null | undefined, d = 4) =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(d);
const sign = (v: number | null | undefined, d = 4) =>
  v == null || Number.isNaN(v) ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(d)}`;

const TABS = [
  { key: "selection", icon: GitCompareArrows },
  { key: "parity", icon: LineChartIcon },
  { key: "hypotheses", icon: Activity },
  { key: "sobol", icon: Layers },
] as const;

type TabKey = (typeof TABS)[number]["key"];

function Verdict({ value }: { value: string }) {
  const ok = value === VERDICT_OK;
  const unknown = value !== VERDICT_OK && value !== VERDICT_NO;
  const color = unknown ? "#94a3b8" : ok ? "#10b981" : "#ef4444";
  return (
    <span className="ml-eval-verdict" style={{ color, borderColor: `${color}55`, background: `${color}14` }}>
      {unknown ? value : ok ? "✓" : "✕"} {value}
    </span>
  );
}

export default function MlEvaluation({ data }: { data: any }) {
  const { t } = useTranslation();
  const [tab, setTab] = useState<TabKey>("selection");

  if (!data || data.available === false || !data.model_selection) {
    return (
      <div className="card">
        <p style={{ margin: 0, color: "var(--text-muted)" }}>{t("crispdm.mlEval.empty")}</p>
      </div>
    );
  }

  const sel = data.model_selection;
  const par = data.parity ?? {};
  const hyp = data.hypotheses?.hypotheses ?? [];
  const sob = data.sobol ?? {};
  const paired = sel.paired_test ?? {};
  const currentTab = TABS.find((x) => x.key === tab)!;

  return (
    <div className="ml-eval">
      <div className="ml-eval-tabs" role="tablist">
        {TABS.map(({ key, icon: Icon }) => (
          <button
            key={key}
            role="tab"
            aria-selected={tab === key}
            title={t(`crispdm.mlEval.tabs.${key}.question`)}
            className={`ml-eval-tab${tab === key ? " active" : ""}`}
            onClick={() => setTab(key)}
          >
            <Icon size={14} className="ml-eval-tab-icon" />
            <span className="ml-eval-tab-text">
              <b>{t(`crispdm.mlEval.tabs.${key}.title`)}</b>
              <em>{t(`crispdm.mlEval.tabs.${key}.question`)}</em>
            </span>
          </button>
        ))}
      </div>

      <p className="ml-eval-tab-question">
        {t(`crispdm.mlEval.tabs.${currentTab.key}.question`)}
      </p>

      {tab === "selection" && (
        <div role="tabpanel">
          <div className="grid grid-4" style={{ marginBottom: 16 }}>
            <StatCard
              label={t("crispdm.mlEval.deployed")}
              value={sel.deployed}
              subtext={`${sel.n_train} / ${sel.n_test} ${t("crispdm.mlEval.trainTest")}`}
              icon={<GitCompareArrows size={22} style={{ color: "#fbbf24" }} />}
            />
            <StatCard
              label={t("crispdm.mlEval.selectionMetric")}
              value={sel.cv_f1 != null ? num(sel.cv_f1) : sel.deployed}
              subtext={`${sel.selection_metric} · ${sel.cv_folds} ${t("crispdm.mlEval.folds")}`}
              icon={<Gauge size={22} style={{ color: "#60a5fa" }} />}
            />
            <StatCard
              label={t("crispdm.mlEval.separability")}
              value={paired.separable ? t("crispdm.mlEval.yes") : t("crispdm.mlEval.no")}
              subtext={paired.applicable ? `p = ${num(paired.p_value)}` : t("crispdm.mlEval.notApplicable")}
              icon={
                paired.separable ? (
                  <CheckCircle2 size={22} style={{ color: "#10b981" }} />
                ) : (
                  <XCircle size={22} style={{ color: "#ef4444" }} />
                )
              }
            />
            <StatCard
              label={t("crispdm.mlEval.regressor")}
              value={sel.regressor?.name ?? "—"}
              subtext={`R² ${num(sel.regressor?.r2_test)} · MAE ${num(sel.regressor?.mae_test, 3)}`}
              icon={<LineChartIcon size={22} style={{ color: "#a78bfa" }} />}
            />
          </div>

          <p className="ml-eval-note">{sel.verdict}</p>

          <h4 className="ml-eval-subtitle">{t("crispdm.mlEval.candidates")}</h4>
          <div className="table-responsive">
            <table className="table">
              <thead>
                <tr>
                  <th>{t("crispdm.mlEval.candidate")}</th>
                  <th>{t("crispdm.mlEval.role")}</th>
                  <th>CV F1 μ ± σ</th>
                  <th>F1 {t("crispdm.mlEval.test")}</th>
                  <th>{t("crispdm.mlEval.accuracy")}</th>
                  <th>ROC AUC</th>
                  <th>s</th>
                </tr>
              </thead>
              <tbody>
                {(sel.candidates ?? []).map((c: any) => (
                  <tr key={c.name} style={c.selected ? { background: "rgba(251,191,36,0.07)" } : undefined}>
                    <td>
                      <b>{c.name}</b>
                      {c.selected && <span className="ml-eval-flag">{t("crispdm.mlEval.selected")}</span>}
                    </td>
                    <td style={{ color: "var(--text-muted)" }}>{c.role}</td>
                    <td style={{ fontFamily: "var(--font-mono)" }}>{num(c.cv_f1_mean)} ± {num(c.cv_f_std)}</td>
                    <td style={{ fontFamily: "var(--font-mono)" }}>{num(c.test_f1_macro)}</td>
                    <td style={{ fontFamily: "var(--font-mono)" }}>{num(c.test_accuracy)}</td>
                    <td style={{ fontFamily: "var(--font-mono)" }}>{num(c.test_roc_auc)}</td>
                    <td style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>{num(c.train_seconds, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {paired.applicable && (
            <>
              <h4 className="ml-eval-subtitle">{t("crispdm.mlEval.paired")}</h4>
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={(paired.winner_scores ?? []).map((w: number, i: number) => ({ fold: i + 1, winner: w, runner: paired.runner_scores?.[i] }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
                  <XAxis dataKey="fold" {...AXIS} />
                  <YAxis domain={[0.6, 1]} {...AXIS} />
                  <Tooltip {...TOOLTIP} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="winner" name={sel.deployed} fill="#fbbf24" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="runner" name={t("crispdm.mlEval.runner")} fill="#475569" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <p className="ml-eval-note">
                {t("crispdm.mlEval.meanDiff")} <b>{sign(paired.mean_diff)}</b> ·{" "}
                {t("crispdm.mlEval.wilcoxon")} W = {num(paired.wilcoxon_stat, 1)}, p = {num(paired.p_value)} ·{" "}
                {t("crispdm.mlEval.nSamples")} {sel.n_samples}
              </p>
            </>
          )}
        </div>
      )}

      {tab === "parity" && (
        <div role="tabpanel">
          <div className="grid grid-4" style={{ marginBottom: 16 }}>
            <StatCard label="R²" value={num(par.r2_test)} subtext={par.regressor} icon={<LineChartIcon size={22} style={{ color: "#10b981" }} />} />
            <StatCard label="MAE" value={num(par.mae_test, 3)} subtext={t("crispdm.mlEval.pp")} icon={<Gauge size={22} style={{ color: "#60a5fa" }} />} />
            <StatCard label="RMSE" value={num(par.rmse_test, 3)} subtext={`σ ${num(par.resid_sd, 3)}`} icon={<Gauge size={22} style={{ color: "#a78bfa" }} />} />
            <StatCard
              label={t("crispdm.mlEval.levelAgreement")}
              value={pct(par.level_agreement, 1)}
              subtext={t("crispdm.mlEval.observedVsPredicted")}
              icon={<CheckCircle2 size={22} style={{ color: "#fb923c" }} />}
            />
          </div>

          <div className="grid grid-2" style={{ marginBottom: 18 }}>
            <div>
              <h4 className="ml-eval-subtitle">{t("crispdm.mlEval.parityPlot")}</h4>
              <ResponsiveContainer width="100%" height={280}>
                <ScatterChart margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
                  <XAxis type="number" dataKey="x" name={t("crispdm.mlEval.observed")} {...AXIS} />
                  <YAxis type="number" dataKey="y" name={t("crispdm.mlEval.predicted")} {...AXIS} />
                  <ZAxis range={[26, 26]} />
                  <Tooltip {...TOOLTIP} cursor={{ strokeDasharray: "3 3" }} />
                  <ReferenceLine
                    segment={[
                      { x: par.obs_min ?? 0, y: par.obs_min ?? 0 },
                      { x: par.obs_max ?? 100, y: par.obs_max ?? 100 },
                    ]}
                    stroke="#475569"
                    strokeDasharray="4 4"
                  />
                  <Scatter data={par.parity_points ?? []} fill="#3b82f6" fillOpacity={0.55} />
                </ScatterChart>
              </ResponsiveContainer>
              <p className="ml-eval-note">
                {t("crispdm.mlEval.fit")} y = {num(par.fit_line?.slope, 3)} x {par.fit_line?.intercept >= 0 ? "+" : "−"} {num(Math.abs(par.fit_line?.intercept ?? 0), 3)} ·{" "}
                {t("crispdm.mlEval.interceptP")} p = {num(par.fit_line?.p_value)} ·{" "}
                {t("crispdm.mlEval.bias")} {sign(par.bias, 4)}
              </p>
            </div>

            <div>
              <h4 className="ml-eval-subtitle">{t("crispdm.mlEval.residualHist")}</h4>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={((par.residual_hist?.counts ?? []) as number[]).map((c, i) => ({ bin: i, count: c }))} margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
                  <XAxis dataKey="bin" {...AXIS} />
                  <YAxis {...AXIS} />
                  <Tooltip {...TOOLTIP} />
                  <Bar dataKey="count" fill="#64748b" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <p className="ml-eval-note">
                {par.normality?.test} p = {num(par.normality?.p_value)} ·{" "}
                {par.normality?.normal_at_05 ? t("crispdm.mlEval.notRejected") : t("crispdm.mlEval.rejected")} ·{" "}
                {par.heteroscedasticity?.applicable && (
                  <>
                    {t("crispdm.mlEval.bp")} p = {num(par.heteroscedasticity?.p_value)} ·{" "}
                    {par.heteroscedasticity?.homoscedastic ? t("crispdm.mlEval.homoscedastic") : t("crispdm.mlEval.heteroscedastic")}
                  </>
                )}
              </p>
            </div>
          </div>

          <h4 className="ml-eval-subtitle">{t("crispdm.mlEval.residualVsFitted")}</h4>
          <ResponsiveContainer width="100%" height={220}>
            <ScatterChart margin={{ top: 8, right: 12, bottom: 4, left: -18 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
              <XAxis type="number" dataKey="x" name={t("crispdm.mlEval.fitted")} {...AXIS} />
              <YAxis type="number" dataKey="y" name={t("crispdm.mlEval.residual")} {...AXIS} />
              <ZAxis range={[22, 22]} />
              <Tooltip {...TOOLTIP} cursor={{ strokeDasharray: "3 3" }} />
              <ReferenceLine y={0} stroke="#ef4444" strokeDasharray="4 4" />
              <Scatter data={par.residual_vs_fitted ?? []} fill="#a78bfa" fillOpacity={0.5} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {tab === "hypotheses" && (
        <div role="tabpanel">
          {hyp.map((h: any) => (
            <div key={h.code} className="card ml-eval-hyp">
              <div className="ml-eval-hyp-head">
                <b>{h.code}</b>
                <span>{h.statement}</span>
                <Verdict value={h.verdict} />
              </div>
              <dl className="ml-eval-hyp-body">
                <dt>{t("crispdm.mlEval.criterion")}</dt>
                <dd>{h.criterion}</dd>
                <dt>{t("crispdm.mlEval.observed")}</dt>
                <dd>{h.observed}</dd>
                {h.verdict_reason && (
                  <>
                    <dt>{t("crispdm.mlEval.reason")}</dt>
                    <dd>{h.verdict_reason}</dd>
                  </>
                )}
              </dl>
            </div>
          ))}
          {data.hypotheses?.method_note && <p className="ml-eval-note">{data.hypotheses.method_note}</p>}
        </div>
      )}

      {tab === "sobol" && (
        <div role="tabpanel">
          <div className="grid grid-4" style={{ marginBottom: 16 }}>
            <StatCard label={t("crispdm.mlEval.varExplained")} value={pct(sob.var_model != null ? 1 - (sob.unexplained_share ?? 0) : null, 1)} subtext={sob.method} icon={<Layers size={22} style={{ color: "#a78bfa" }} />} />
            <StatCard label={t("crispdm.mlEval.conditionNumber")} value={num(sob.collinearity?.condition_number, 0)} subtext={t("crispdm.mlEval.ofPredictors")} icon={<Activity size={22} style={{ color: "#ef4444" }} />} />
            <StatCard label={t("crispdm.mlEval.maxRho")} value={num(sob.collinearity?.max_abs_rho)} subtext={`${sob.collinearity?.pairs_above_0_7} / ${sob.collinearity?.pairs} > 0,7`} icon={<GitCompareArrows size={22} style={{ color: "#fb923c" }} />} />
            <StatCard label="VIF" value={num(sob.collinearity?.vif_max, 0)} subtext={`mediana ${num(sob.collinearity?.vif_median, 0)}`} icon={<Layers size={22} style={{ color: "#60a5fa" }} />} />
          </div>

          <p className="ml-eval-note">
            {t("crispdm.mlEval.sobolMethod")} S1 = σ/τ² y ST = 1 − σ²/τ², donde τ² es la
            varianza total explicada. Con predictores colineales el método de Jansen es
            inestable, por lo que se reporta además la contribución única de cada variable
            una vez controladas las demás.
          </p>

          <div className="grid grid-2">
            <div>
              <h4 className="ml-eval-subtitle">{t("crispdm.mlEval.uniqueShare")}</h4>
              <ResponsiveContainer width="100%" height={Math.max(240, (sob.variables?.length ?? 6) * 26)}>
                <BarChart data={[...(sob.variables ?? [])].sort((a: any, b: any) => b.unique - a.unique)} margin={{ top: 8, right: 12, bottom: 4, left: -18 }} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
                  <XAxis type="number" domain={[0, Math.max(0.05, ...(sob.variables ?? []).map((v: any) => v.unique ?? 0))]} {...AXIS} />
                  <YAxis type="category" dataKey="name" width={150} {...AXIS} />
                  <Tooltip {...TOOLTIP} />
                  <Bar dataKey="unique" radius={[0, 4, 4, 0]}>
                    {[...(sob.variables ?? [])].sort((a: any, b: any) => b.unique - a.unique).map((v: any) => (
                      <Cell key={v.code} fill={v.unique > 0.1 ? "#a78bfa" : v.unique > 0.02 ? "#818cf8" : "#475569"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div>
              <h4 className="ml-eval-subtitle">{t("crispdm.mlEval.table")}</h4>
              <div className="table-responsive">
                <table className="table">
                  <thead>
                    <tr>
                      <th>{t("crispdm.mlEval.variable")}</th>
                      <th>S1</th>
                      <th>ST</th>
                      <th>{t("crispdm.mlEval.unique")}</th>
                      <th>β</th>
                      <th>VIF</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(sob.variables ?? []).map((v: any) => (
                      <tr key={v.code}>
                        <td>
                          {v.name}
                          {v.top_partner && (
                            <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                              {t("crispdm.mlEval.collinearWith")} {v.top_partner}
                            </div>
                          )}
                        </td>
                        <td style={{ fontFamily: "var(--font-mono)" }}>{num(v.S1)}</td>
                        <td style={{ fontFamily: "var(--font-mono)" }}>{num(v.ST)}</td>
                        <td style={{ fontFamily: "var(--font-mono)" }}>{num(v.unique)}</td>
                        <td style={{ fontFamily: "var(--font-mono)" }}>{sign(v.coefficient)}</td>
                        <td style={{ fontFamily: "var(--font-mono)", color: v.vif > 10 ? "#ef4444" : undefined }}>{num(v.vif, 0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
