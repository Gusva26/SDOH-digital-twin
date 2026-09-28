import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Award, Cpu, Database, Gauge, PlayCircle, Target } from "lucide-react";
import api from "../api/client";
import { errorMessage } from "../api/errors";
import StatCard from "../components/StatCard";

const RISK_LEVELS = ["low", "moderate", "high", "critical"] as const;
const ACCENT = "#fbbf24";
const MUTED_BAR = "#64748b";

const num = (v: unknown, digits = 4) =>
  typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: digits }) : "—";

const tooltipStyle = {
  background: "var(--bg-surface)",
  border: "1px solid rgba(59,130,246,0.3)",
  borderRadius: 8,
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card" style={{ marginBottom: 22 }}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      {children}
    </div>
  );
}

export default function MlModels() {
  const { t } = useTranslation();
  const [status, setStatus] = useState<any>(null);
  const [preds, setPreds] = useState<any>(null);
  const [target, setTarget] = useState("");
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<any>(null);
  const [predicting, setPredicting] = useState(false);
  const [levelFilter, setLevelFilter] = useState("");

  const loadStatus = useCallback(() => {
    api
      .get("/ml/status")
      .then((r) => {
        setStatus(r.data);
        setTarget((cur) => cur || r.data.target?.code || r.data.default_target);
        if (r.data.trained) {
          const med = r.data.feature_medians ?? {};
          setInputs(Object.fromEntries(Object.entries(med).map(([k, v]) => [k, String(v)])));
          api.get("/ml/predictions").then((p) => setPreds(p.data)).catch(() => setPreds(null));
        }
      })
      .catch((e) => setError(errorMessage(e, t("ml.loadError"))));
  }, [t]);

  useEffect(loadStatus, [loadStatus]);

  const train = async () => {
    setTraining(true);
    setError(null);
    setPrediction(null);
    try {
      await api.post(`/ml/train?target=${encodeURIComponent(target)}`, null, { timeout: 600_000 });
      loadStatus();
    } catch (e) {
      setError(errorMessage(e, t("ml.trainError")));
    } finally {
      setTraining(false);
    }
  };

  const predict = async () => {
    setPredicting(true);
    setError(null);
    try {
      const features = Object.fromEntries(
        Object.entries(inputs).map(([k, v]) => [k, v === "" ? null : Number(v)])
      );
      const r = await api.post("/ml/predict", { features });
      setPrediction(r.data);
    } catch (e) {
      setError(errorMessage(e, t("ml.predictError")));
    } finally {
      setPredicting(false);
    }
  };

  const models: any[] = status?.models ?? [];
  const best = models.find((m) => m.selected);
  const featureNames: Record<string, string> = useMemo(
    () => Object.fromEntries((status?.features ?? []).map((f: any) => [f.code, f.name])),
    [status]
  );
  const tractRows = useMemo(
    () => (preds?.items ?? []).filter((i: any) => !levelFilter || i.risk_level === levelFilter),
    [preds, levelFilter]
  );

  return (
    <div>
      <div className="topbar">
        <div className="page-title-group">
          <h2 className="page-title">{t("ml.title")}</h2>
          <p className="page-subtitle">{t("ml.subtitle")}</p>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 13, color: "var(--text-muted)", fontWeight: 600 }}>
              {t("ml.target")}:
            </span>
            <select className="input" value={target} onChange={(e) => setTarget(e.target.value)}>
              {(status?.targets ?? []).map((code: string) => (
                <option key={code} value={code}>{t(`ml.targets.${code}`, code)}</option>
              ))}
            </select>
          </div>
          <button className="btn" onClick={train} disabled={training || !target}>
            <PlayCircle size={16} />
            <span>{training ? t("ml.training") : t("ml.train")}</span>
          </button>
        </div>
      </div>

      {error && <div className="card" style={{ marginBottom: 22 }}><span className="error-text">{error}</span></div>}
      {training && (
        <div className="card" style={{ marginBottom: 22 }}>
          <span style={{ color: "var(--text-muted)" }}>{t("ml.trainingHint")}</span>
        </div>
      )}

      {status && !status.trained && !training && (
        <div className="card" style={{ marginBottom: 22 }}>
          <p style={{ margin: 0, color: "var(--text-muted)" }}>{t("ml.notTrained")}</p>
        </div>
      )}

      {status?.trained && best && (
        <>
          <div className="grid grid-4" style={{ marginBottom: 22 }}>
            <StatCard label={t("ml.deployedModel")} value={status.best_model_name} subtext={t("ml.selectedBy")} icon={<Award size={22} style={{ color: ACCENT }} />} />
            <StatCard label={t("ml.f1cv")} value={num(best.cv_f1_mean)} subtext={`± ${num(best.cv_f1_std, 3)} · k=${status.cv_folds}`} icon={<Target size={22} style={{ color: "#60a5fa" }} />} />
            <StatCard label={t("ml.aucTest")} value={num(best.test_roc_auc)} subtext={`F1 test ${num(best.test_f1_macro)}`} icon={<Gauge size={22} style={{ color: "#fb923c" }} />} />
            <StatCard label={t("ml.samples")} value={status.n_samples} subtext={`train ${status.n_train} · test ${status.n_test}`} icon={<Database size={22} style={{ color: "#34d399" }} />} />
          </div>

          <Section title={t("ml.dataset")}>
            <p style={{ color: "var(--text-muted)", lineHeight: 1.6, margin: 0 }}>
              <b>{t("ml.target")}:</b> {status.target?.name} ({status.year}) · {t("ml.classesByQuartile")}{" "}
              <span style={{ fontFamily: "var(--font-mono)" }}>
                moderate ≥ {num(status.class_cuts?.moderate, 2)} · high ≥ {num(status.class_cuts?.high, 2)} · critical ≥ {num(status.class_cuts?.critical, 2)}
              </span>
              <br />
              <b>{t("ml.features")}:</b> {(status.features ?? []).map((f: any) => f.name).join(" · ")}
              <br />
              <span style={{ color: "var(--text-dim)", fontSize: 12.5 }}>
                {status.source} · {t("ml.trainedAt")} {status.trained_at?.replace("T", " ").replace("Z", "")} UTC
              </span>
            </p>
          </Section>

          <Section title={t("ml.comparison")}>
            <div className="table-responsive">
              <table className="table">
                <thead>
                  <tr>
                    <th>{t("ml.model")}</th><th>{t("ml.bestParams")}</th><th>F1 CV</th>
                    <th>F1 test</th><th>Accuracy</th><th>ROC-AUC</th><th>{t("ml.time")}</th><th>{t("common.status")}</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => (
                    <tr key={m.key}>
                      <td>
                        <b>{m.name}</b>
                        <div style={{ fontSize: 11.5, color: "var(--text-dim)" }}>{m.role}</div>
                      </td>
                      <td style={{ fontFamily: "var(--font-mono)", fontSize: 11.5, color: "var(--text-muted)" }}>
                        {Object.entries(m.best_params).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")}
                      </td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(m.cv_f1_mean)} ± {num(m.cv_f1_std, 3)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(m.test_f1_macro)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(m.test_accuracy)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(m.test_roc_auc)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(m.train_seconds, 1)} s</td>
                      <td>
                        <span className={`badge ${m.selected ? "green" : "gray"}`}>
                          {m.selected ? t("ml.deployed") : t("ml.candidate")}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <div className="grid grid-2">
            <Section title={t("ml.f1Chart")}>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={models.map((m) => ({ name: m.name, f1: m.cv_f1_mean, selected: m.selected }))} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                  <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 11 }} interval={0} />
                  <YAxis stroke="#64748b" domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} contentStyle={tooltipStyle} formatter={(v: number) => [num(v), "F1-macro CV"]} />
                  <Bar dataKey="f1" radius={[4, 4, 0, 0]} maxBarSize={56}>
                    {models.map((m) => (
                      <Cell key={m.key} fill={m.selected ? ACCENT : MUTED_BAR} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Section>

            <Section title={`${t("ml.confusion")} · ${best.name}`}>
              <ConfusionMatrix matrix={best.confusion_matrix} t={t} />
            </Section>
          </div>

          <Section title={t("ml.importance")}>
            <ResponsiveContainer width="100%" height={Math.max(220, (status.feature_importance?.length ?? 0) * 28)}>
              <BarChart layout="vertical" data={status.feature_importance} margin={{ top: 0, right: 20, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" horizontal={false} />
                <XAxis type="number" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" width={230} stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} contentStyle={tooltipStyle} formatter={(v: number) => [num(v), t("ml.importanceUnit")]} />
                <Bar dataKey="importance" fill="#60a5fa" radius={[0, 4, 4, 0]} maxBarSize={18} />
              </BarChart>
            </ResponsiveContainer>
            <p style={{ color: "var(--text-dim)", fontSize: 12.5, marginBottom: 0 }}>{t("ml.importanceHint")}</p>
          </Section>

          <Section title={t("ml.simulator")}>
            <p style={{ color: "var(--text-muted)", marginTop: 0, fontSize: 13.5 }}>{t("ml.simulatorHint")}</p>
            <div className="grid grid-4" style={{ gap: 12, marginBottom: 14 }}>
              {Object.keys(inputs).map((code) => (
                <label key={code} style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12, color: "var(--text-muted)" }}>
                  {featureNames[code] ?? code}
                  <input
                    className="input"
                    type="number"
                    step="0.1"
                    value={inputs[code]}
                    onChange={(e) => setInputs((s) => ({ ...s, [code]: e.target.value }))}
                  />
                </label>
              ))}
            </div>
            <button className="btn" onClick={predict} disabled={predicting}>
              <Cpu size={16} />
              <span>{predicting ? t("ml.predicting") : t("ml.predict")}</span>
            </button>
            {prediction && (
              <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 12 }}>
                <div style={{ display: "flex", gap: 18, alignItems: "center", flexWrap: "wrap" }}>
                  <div>
                    <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{t("ml.predictedLevel")}</div>
                    <span className={`badge ${prediction.risk_level}`} style={{ fontSize: 14 }}>
                      {t(`ml.levels.${prediction.risk_level}`)}
                    </span>
                  </div>
                  {prediction.estimated_value != null && (
                    <div>
                      <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{t("ml.estimatedValue")}</div>
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: 14, fontWeight: 600 }}>
                        {num(prediction.estimated_value, 2)} %
                      </span>
                    </div>
                  )}
                  <div>
                    <div style={{ fontSize: 12, color: "var(--text-muted)" }}>{t("ml.classifierVote")}</div>
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: 13 }}>
                      {prediction.classifier_level_name
                        ? `${t(`ml.levels.${prediction.classifier_level}`)} · ${num(prediction.confidence * 100, 0)} %`
                        : `${num(prediction.confidence * 100, 1)} %`}
                    </span>
                  </div>
                </div>
                {prediction.estimated_value != null && (
                  <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                    {t("ml.estimateInterval")}: {num(prediction.estimate_interval?.[0], 2)} – {num(prediction.estimate_interval?.[1], 2)} %
                    {" · "}
                    {t(`ml.${prediction.level_range?.open_min && prediction.level_range?.open_max
                      ? "bandOpenMin"
                      : prediction.level_range?.open_min
                        ? "bandOpenMin"
                        : prediction.level_range?.open_max
                          ? "bandOpenMax"
                          : "bandClosed"}`, {
                      min: num(prediction.level_range?.min, 1),
                      max: num(prediction.level_range?.max, 1),
                    })}
                  </div>
                )}
                {prediction.coherence === "discrepancy" && (
                  <div
                    style={{
                      fontSize: 12.5,
                      lineHeight: 1.5,
                      color: "var(--text-muted)",
                      borderLeft: "3px solid var(--border-subtle)",
                      paddingLeft: 10,
                    }}
                  >
                    {t("ml.discrepancyWarn", {
                      level: t(`ml.levels.${prediction.classifier_level}`),
                      conf: num(prediction.confidence * 100, 0),
                      gap: num(prediction.cut_gap_pp, 2),
                    })}
                  </div>
                )}
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  {RISK_LEVELS.map((l) => (
                    <span key={l} style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                      {t(`ml.levels.${l}`)}: {num((prediction.probabilities?.[l] ?? 0) * 100, 1)} %
                    </span>
                  ))}
                </div>
              </div>
            )}
          </Section>

          {preds && (
            <Section title={t("ml.tractPredictions")}>
              <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
                {RISK_LEVELS.map((l) => (
                  <button
                    key={l}
                    type="button"
                    className={`badge ${levelFilter === l ? l : "gray"}`}
                    style={{ cursor: "pointer" }}
                    onClick={() => setLevelFilter((cur) => (cur === l ? "" : l))}
                  >
                    {t(`ml.levels.${l}`)} · {preds.summary?.[l] ?? 0}
                  </button>
                ))}
                <span style={{ marginLeft: "auto", fontSize: 13, color: "var(--text-muted)", display: "flex", gap: 14, flexWrap: "wrap" }}>
                  {preds.coherence_pct != null && (
                    <span>{t("ml.levelAgreement")}: <b>{num(preds.coherence_pct, 1)} %</b></span>
                  )}
                  <span>{t("ml.agreement")}: <b>{num(preds.agreement_pct, 2)} %</b></span>
                </span>
              </div>
              <div className="table-responsive" style={{ maxHeight: 420, overflowY: "auto" }}>
                <table className="table">
                  <thead>
                    <tr>
                      <th>GEOID</th><th>{t("ml.observed")}</th><th>{t("ml.observedLevel")}</th>
                      <th>{t("ml.predictedLevel")}</th><th>{t("ml.classifierVoteCol")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tractRows.slice(0, 300).map((i: any) => (
                      <tr key={i.geoid}>
                        <td style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>{i.geoid}</td>
                        <td style={{ fontFamily: "var(--font-mono)" }}>{num(i.observed_value, 2)}</td>
                        <td>{i.observed_level ? <span className={`badge ${i.observed_level}`}>{t(`ml.levels.${i.observed_level}`)}</span> : "—"}</td>
                        <td>
                          <span className={`badge ${i.risk_level}`}>{t(`ml.levels.${i.risk_level}`)}</span>
                          {i.classifier_level && i.classifier_level !== i.risk_level && (
                            <span style={{ marginLeft: 6, fontSize: 11, color: "var(--text-muted)" }}>
                              ({t(`ml.levels.${i.classifier_level}`)})
                            </span>
                          )}
                        </td>
                        <td style={{ fontFamily: "var(--font-mono)" }}>{num(i.confidence * 100, 1)} %</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {tractRows.length > 300 && (
                <p style={{ color: "var(--text-dim)", fontSize: 12.5, marginBottom: 0 }}>
                  {t("ml.showingFirst", { shown: 300, total: tractRows.length })}
                </p>
              )}
            </Section>
          )}

          <Section title={t("ml.limitations")}>
            <ul style={{ margin: 0, paddingLeft: 18, display: "flex", flexDirection: "column", gap: 7 }}>
              {[t("ml.limit1"), t("ml.limit2"), t("ml.limit3")].map((l) => (
                <li key={l} style={{ color: "var(--text-muted)", fontSize: 13.5, lineHeight: 1.55 }}>{l}</li>
              ))}
            </ul>
          </Section>
        </>
      )}
    </div>
  );
}

function ConfusionMatrix({ matrix, t }: { matrix: number[][]; t: (k: string) => string }) {
  const max = Math.max(1, ...matrix.flat());
  return (
    <div className="table-responsive">
      <table className="table" style={{ textAlign: "center" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", fontSize: 11 }}>{t("ml.realVsPred")}</th>
            {RISK_LEVELS.map((l) => <th key={l} style={{ textAlign: "center" }}>{t(`ml.levels.${l}`)}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={RISK_LEVELS[i]}>
              <td style={{ textAlign: "left", fontWeight: 600 }}>{t(`ml.levels.${RISK_LEVELS[i]}`)}</td>
              {row.map((v, j) => (
                <td
                  key={j}
                  title={`${t(`ml.levels.${RISK_LEVELS[i]}`)} → ${t(`ml.levels.${RISK_LEVELS[j]}`)}: ${v}`}
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontWeight: i === j ? 700 : 400,
                    background: `rgba(59,130,246,${(0.08 + 0.55 * (v / max)).toFixed(2)})`,
                  }}
                >
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
