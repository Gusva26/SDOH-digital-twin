import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
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
import {
  Brain,
  CheckCircle2,
  Cpu,
  ClipboardCheck,
  Database,
  Gauge,
  PlayCircle,
  Rocket,
  Target,
  Wand2,
  XCircle,
} from "lucide-react";
import api from "../api/client";
import { errorMessage } from "../api/errors";
import StatCard from "../components/StatCard";
import MlEvaluation from "../components/MlEvaluation";
import type { CrispPhase, CrispPhaseOverview } from "../types";

const PHASE_META: Record<
  string,
  { icon: typeof Target; endpoint: string; color: string }
> = {
  "business-understanding": { icon: Target, endpoint: "/crispdm/business-understanding", color: "#60a5fa" },
  "data-understanding": { icon: Database, endpoint: "/crispdm/data-understanding", color: "#34d399" },
  "data-preparation": { icon: Wand2, endpoint: "/crispdm/data-preparation", color: "#a78bfa" },
  modeling: { icon: Brain, endpoint: "/crispdm/modeling", color: "#fbbf24" },
  evaluation: { icon: ClipboardCheck, endpoint: "/crispdm/evaluation", color: "#fb923c" },
  deployment: { icon: Rocket, endpoint: "/crispdm/deployment", color: "#f472b6" },
};

const RISK_COLORS: Record<string, string> = {
  low: "#10b981",
  moderate: "#f59e0b",
  high: "#f97316",
  critical: "#ef4444",
};

const num = (v: unknown, digits = 2) =>
  typeof v === "number" ? v.toLocaleString(undefined, { maximumFractionDigits: digits }) : "—";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card" style={{ marginBottom: 22 }}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      {children}
    </div>
  );
}

function KeyValue({ items }: { items: [string, React.ReactNode][] }) {
  return (
    <div className="grid grid-2" style={{ gap: 10 }}>
      {items.map(([k, v]) => (
        <div key={k} style={{ display: "flex", justifyContent: "space-between", gap: 12, padding: "7px 0", borderBottom: "1px solid var(--border-subtle)" }}>
          <span style={{ color: "var(--text-muted)", fontSize: 13 }}>{k}</span>
          <span style={{ fontWeight: 600, fontFamily: "var(--font-mono)", textAlign: "right" }}>{v}</span>
        </div>
      ))}
    </div>
  );
}

function Bullets({ items }: { items: string[] }) {
  return (
    <ul style={{ margin: 0, paddingLeft: 18, display: "flex", flexDirection: "column", gap: 7 }}>
      {items.map((i) => (
        <li key={i} style={{ color: "var(--text-muted)", fontSize: 13.5, lineHeight: 1.55 }}>{i}</li>
      ))}
    </ul>
  );
}

/** Resumen del modelo ML desplegado, compartido por las fases IV, V y VI. */
function MlSummary({ phase }: { phase: "modeling" | "evaluation" | "deployment" }) {
  const { t } = useTranslation();
  const [ml, setMl] = useState<any>(null);

  useEffect(() => {
    api.get("/ml/status").then((r) => setMl(r.data)).catch(() => setMl(null));
  }, []);

  if (!ml) return null;
  const best = (ml.models ?? []).find((m: any) => m.selected);

  return (
    <Section title={t("crispdm.ml.title")}>
      {!ml.trained || !best ? (
        <p style={{ color: "var(--text-muted)", marginTop: 0 }}>{t("crispdm.ml.none")}</p>
      ) : phase === "modeling" ? (
        <Bullets
          items={(ml.models ?? []).map(
            (m: any) => `${m.name} — ${m.role}${m.selected ? ` ✓ ${t("ml.deployed")}` : ""}`
          )}
        />
      ) : phase === "evaluation" ? (
        <div className="table-responsive">
          <table className="table">
            <thead>
              <tr><th>{t("ml.model")}</th><th>F1 CV</th><th>F1 test</th><th>ROC-AUC</th><th>{t("common.status")}</th></tr>
            </thead>
            <tbody>
              {ml.models.map((m: any) => (
                <tr key={m.key}>
                  <td><b>{m.name}</b></td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{num(m.cv_f1_mean, 4)}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{num(m.test_f1_macro, 4)}</td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>{num(m.test_roc_auc, 4)}</td>
                  <td><span className={`badge ${m.selected ? "green" : "gray"}`}>{m.selected ? t("ml.deployed") : t("ml.candidate")}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <KeyValue
          items={[
            [t("ml.deployedModel"), ml.best_model_name],
            [t("ml.target"), ml.target?.name],
            ["F1 CV", num(best.cv_f1_mean, 4)],
            ["Endpoint", "POST /api/ml/predict"],
          ]}
        />
      )}
      {ml.trained && (
        <p style={{ color: "var(--text-dim)", fontSize: 12.5, marginBottom: 0 }}>
          {ml.selection_metric} · n={ml.n_samples} · {ml.target?.name}
        </p>
      )}
      <Link to="/modelos-ml" className="btn" style={{ marginTop: 12, display: "inline-flex", textDecoration: "none" }}>
        <Cpu size={16} />
        <span>{t("crispdm.ml.open")}</span>
      </Link>
    </Section>
  );
}

export default function CrispDm() {
  const { t } = useTranslation();
  const { phase } = useParams<{ phase?: string }>();
  const [overview, setOverview] = useState<CrispPhaseOverview | null>(null);
  const [data, setData] = useState<any>(null);
  const [loadedPhase, setLoadedPhase] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [year, setYear] = useState<number | "">("");
  const [pipelineResult, setPipelineResult] = useState<any>(null);
  const [mlEval, setMlEval] = useState<any>(null);
  const [mlEvalTarget, setMlEvalTarget] = useState<string>("");
  const [mlEvalRunning, setMlEvalRunning] = useState(false);

  const activeKey = phase && PHASE_META[phase] ? phase : null;
  const meta = activeKey ? PHASE_META[activeKey] : null;

  // Los datos solo son válidos para la vista si coinciden con la fase activa en la URL
  const phaseData = loadedPhase === activeKey ? data : null;

  const loadOverview = useCallback(() => {
    api.get("/crispdm/phases").then((r) => {
      setOverview(r.data);
      setYear((y) => (y === "" ? r.data.year : y));
    }).catch(() => {});
  }, []);

  useEffect(loadOverview, [loadOverview]);

  useEffect(() => {
    if (!meta || !activeKey) {
      setData(null);
      setLoadedPhase(null);
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    const targetPhase = activeKey;
    const q = year === "" || targetPhase === "business-understanding" || targetPhase === "deployment"
      ? "" : `?year=${year}`;

    api.get(`${meta.endpoint}${q}`)
      .then((r) => {
        if (!cancelled) {
          setData(r.data);
          setLoadedPhase(targetPhase);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setError(errorMessage(e, t("crispdm.loadError")));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [meta, activeKey, year, t]);

  const loadPhase = useCallback(() => {
    if (!meta || !activeKey) return;
    setLoading(true);
    setError(null);
    const targetPhase = activeKey;
    const q = year === "" || targetPhase === "business-understanding" || targetPhase === "deployment"
      ? "" : `?year=${year}`;
    api.get(`${meta.endpoint}${q}`)
      .then((r) => {
        setData(r.data);
        setLoadedPhase(targetPhase);
      })
      .catch((e) => setError(errorMessage(e, t("crispdm.loadError"))))
      .finally(() => setLoading(false));
  }, [meta, activeKey, year, t]);

  const runPipeline = async () => {
    setRunning(true);
    setError(null);
    setPipelineResult(null);
    try {
      const r = await api.post(`/crispdm/pipeline/run${year === "" ? "" : `?year=${year}`}`, null, { timeout: 600_000 });
      setPipelineResult(r.data);
      loadOverview();
      loadPhase();
    } catch (e: any) {
      setError(errorMessage(e, t("crispdm.runError")));
    } finally {
      setRunning(false);
    }
  };

  const runEvaluation = async () => {
    setRunning(true);
    setError(null);
    try {
      const r = await api.post(`/crispdm/evaluation/run${year === "" ? "" : `?year=${year}`}`);
      setData({ ...r.data, __benchmark: true });
      setLoadedPhase("evaluation");
    } catch (e: any) {
      setError(errorMessage(e, t("crispdm.runError")));
    } finally {
      setRunning(false);
    }
  };

  // Fase V: la evaluación ML persistida se carga aparte del banco de pruebas.
  const loadMlEval = useCallback(() => {
    const q = mlEvalTarget ? `?target=${mlEvalTarget}` : "";
    api.get(`/crispdm/evaluation/ml${q}`).then((r) => setMlEval(r.data)).catch(() => setMlEval(null));
  }, [mlEvalTarget]);

  useEffect(() => {
    if (activeKey === "evaluation") loadMlEval();
  }, [activeKey, loadMlEval]);

  const runMlEval = async () => {
    setMlEvalRunning(true);
    setError(null);
    try {
      const q = mlEvalTarget ? `?target=${mlEvalTarget}` : "";
      const r = await api.post(`/crispdm/evaluation/ml/run${q}`, null, { timeout: 600_000 });
      setMlEval(r.data);
    } catch (e: any) {
      setError(errorMessage(e, t("crispdm.mlEval.error")));
    } finally {
      setMlEvalRunning(false);
    }
  };

  // Fase V: el último banco de pruebas persistido se muestra aunque se recargue la página.
  const bench = phaseData?.latency ? phaseData : phaseData?.last_run ?? null;

  const current: CrispPhase | undefined = useMemo(
    () => overview?.phases.find((p) => p.key === activeKey),
    [overview, activeKey]
  );

  return (
    <div>
      <div className="topbar">
        <div className="page-title-group">
          {activeKey && (
            <Link
              to="/crisp-dm"
              style={{
                fontSize: 12,
                color: "var(--text-muted)",
                textDecoration: "none",
                display: "inline-flex",
                alignItems: "center",
                gap: 4,
                marginBottom: 4,
              }}
            >
              ← {t("crispdm.nav.overview")}
            </Link>
          )}
          <h2 className="page-title">
            {current ? `${current.roman}. ${current.name}` : t("crispdm.title")}
          </h2>
          <p className="page-subtitle">
            {current ? current.question : t("crispdm.subtitle")}
          </p>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 13, color: "var(--text-muted)", fontWeight: 600 }}>
              {t("common.year")}:
            </span>
            <input
              className="input"
              type="number"
              style={{ width: 88 }}
              value={year}
              onChange={(e) => setYear(e.target.value === "" ? "" : +e.target.value)}
            />
          </div>
          {activeKey === "evaluation" && (
            <>
              <select
                className="input"
                style={{ width: 190 }}
                value={mlEvalTarget}
                onChange={(e) => setMlEvalTarget(e.target.value)}
                aria-label="ML target"
              >
                <option value="">
                  {t("crispdm.mlEval.defaultTarget")}
                  {mlEval?.target ? ` — ${t(`ml.targets.${mlEval.target}`, mlEval.target)}` : ""}
                </option>
                {(mlEval?.targets ?? []).map((tc: string) => (
                  <option key={tc} value={tc}>{t(`ml.targets.${tc}`, tc.replace("pct_", ""))}</option>
                ))}
              </select>
              <button className="btn" onClick={runMlEval} disabled={mlEvalRunning || running}>
                <Brain size={16} />
                <span>{mlEvalRunning ? t("crispdm.mlEval.running") : t("crispdm.mlEval.run")}</span>
              </button>
              <button className="btn" onClick={runEvaluation} disabled={running}>
                <Gauge size={16} />
                <span>{running ? t("crispdm.running") : t("crispdm.runBenchmark")}</span>
              </button>
            </>
          )}
          <button className="btn" onClick={runPipeline} disabled={running}>
            <PlayCircle size={16} />
            <span>{running ? t("crispdm.running") : t("crispdm.runPipeline")}</span>
          </button>
        </div>
      </div>

      {/* Stepper interactivo de las seis fases */}
      {overview && (
        <div className="card" style={{ marginBottom: 22, overflowX: "auto" }}>
          <div style={{ display: "flex", gap: 10, minWidth: 720 }}>
            {overview.phases.map((p) => {
              const Icon = PHASE_META[p.key]?.icon ?? Target;
              const isActive = p.key === activeKey;
              return (
                <Link
                  key={p.key}
                  to={`/crisp-dm/${p.key}`}
                  style={{
                    flex: 1,
                    padding: "12px 14px",
                    borderRadius: 10,
                    border: `1px solid ${isActive ? "rgba(59,130,246,0.55)" : "var(--border-subtle)"}`,
                    background: isActive ? "rgba(59,130,246,0.10)" : "transparent",
                    textDecoration: "none",
                    color: "inherit",
                    display: "block",
                    cursor: "pointer",
                    transition: "border-color 0.15s, background-color 0.15s",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                    <Icon size={16} style={{ color: PHASE_META[p.key]?.color }} />
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-dim)" }}>
                      {p.roman}
                    </span>
                    {p.ready ? (
                      <CheckCircle2 size={14} style={{ color: "#10b981", marginLeft: "auto" }} />
                    ) : (
                      <XCircle size={14} style={{ color: "var(--text-dim)", marginLeft: "auto" }} />
                    )}
                  </div>
                  <div style={{ fontSize: 12.5, fontWeight: 600, lineHeight: 1.3 }}>{p.name}</div>
                  <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 3 }}>{p.layer}</div>
                </Link>
              );
            })}
          </div>
        </div>
      )}

      {error && <div className="card" style={{ marginBottom: 22 }}><span className="error-text">{error}</span></div>}
      {loading && <div className="empty-state">{t("common.loading")}</div>}
      {running && (
        <div className="card" style={{ marginBottom: 22 }}>
          <span style={{ color: "var(--text-muted)" }}>{t("crispdm.pipelineHint")}</span>
        </div>
      )}
      {pipelineResult && (
        <Section title={`${t("crispdm.pipelineDone")} · ${num(pipelineResult.total_seconds, 1)} s`}>
          <div className="table-responsive">
            <table className="table">
              <thead><tr><th>{t("crispdm.phase")}</th><th>{t("common.description")}</th><th>{t("ml.time")}</th></tr></thead>
              <tbody>
                {pipelineResult.steps.map((st: any, i: number) => (
                  <tr key={i}>
                    <td>
                      <Link to={`/crisp-dm/${st.phase}`} style={{ color: "inherit", fontWeight: 600, textDecoration: "none" }}>
                        {overview?.phases.find((p) => p.key === st.phase)?.roman}. {overview?.phases.find((p) => p.key === st.phase)?.name ?? st.phase}
                      </Link>
                    </td>
                    <td style={{ color: "var(--text-muted)" }}>{st.summary}</td>
                    <td style={{ fontFamily: "var(--font-mono)" }}>{num(st.seconds, 2)} s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* Vista general */}
      {!activeKey && overview && (
        <>
          <div className="grid grid-4" style={{ marginBottom: 22 }}>
            <StatCard label={t("crispdm.phasesReady")} value={`${overview.completed}/${overview.total}`} icon={<CheckCircle2 size={22} style={{ color: "#10b981" }} />} subtext="CRISP-DM" />
            <StatCard label={t("crispdm.tracts")} value={overview.counts.tracts} icon={<Database size={22} style={{ color: "#34d399" }} />} />
            <StatCard label={t("crispdm.values")} value={overview.counts.values} icon={<Wand2 size={22} style={{ color: "#a78bfa" }} />} />
            <StatCard label={t("crispdm.computedIndexes")} value={overview.counts.computed_indexes} icon={<Brain size={22} style={{ color: "#fbbf24" }} />} />
          </div>
          <Section title={t("crispdm.phaseMap")}>
            <div className="table-responsive">
              <table className="table">
                <thead>
                  <tr>
                    <th>#</th><th>{t("crispdm.phase")}</th><th>{t("crispdm.question")}</th>
                    <th>{t("crispdm.layer")}</th><th>{t("crispdm.objectives")}</th><th>{t("common.status")}</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.phases.map((p) => (
                    <tr key={p.key}>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{p.roman}</td>
                      <td>
                        <Link to={`/crisp-dm/${p.key}`} style={{ color: "inherit", textDecoration: "none", fontWeight: 600 }}>
                          {p.name}
                        </Link>
                      </td>
                      <td style={{ color: "var(--text-muted)" }}>{p.question}</td>
                      <td>{p.layer}</td>
                      <td>{p.objectives.join(", ")}</td>
                      <td>
                        <Link to={`/crisp-dm/${p.key}`} style={{ textDecoration: "none" }}>
                          <span className={`badge ${p.ready ? "green" : "gray"}`}>
                            {p.ready ? t("crispdm.ready") : t("crispdm.pending")}
                          </span>
                        </Link>
                        {p.todo && (
                          <div style={{ fontSize: 11.5, color: "var(--text-dim)", marginTop: 4 }}>{p.todo}</div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </>
      )}

      {/* Fase I */}
      {activeKey === "business-understanding" && phaseData && (
        <>
          <Section title={t("crispdm.goal")}>
            <p style={{ color: "var(--text-muted)", lineHeight: 1.6 }}>{phaseData.goal}</p>
            <p style={{ color: "var(--text-dim)", fontStyle: "italic", lineHeight: 1.6 }}>{phaseData.research_question}</p>
          </Section>
          <div className="grid grid-2">
            <Section title={t("crispdm.objectives")}>
              <Bullets items={(phaseData.objectives ?? []).map((o: any) => `${o.code}: ${o.text}`)} />
            </Section>
            <Section title={t("crispdm.hypotheses")}>
              <Bullets items={(phaseData.hypotheses ?? []).map((h: any) => `${h.code}: ${h.text}`)} />
            </Section>
          </div>
          <div className="grid grid-2">
            <Section title={t("crispdm.successBusiness")}>
              <Bullets items={phaseData.success_criteria?.business ?? []} />
            </Section>
            <Section title={t("crispdm.successDataMining")}>
              <Bullets items={phaseData.success_criteria?.data_mining ?? []} />
            </Section>
          </div>
          <Section title={t("crispdm.stakeholders")}>
            <div className="table-responsive">
              <table className="table">
                <thead>
                  <tr><th>{t("common.name")}</th><th>{t("crispdm.roleCode")}</th><th>{t("crispdm.permissions")}</th><th>{t("crispdm.users")}</th></tr>
                </thead>
                <tbody>
                  {(phaseData.stakeholders ?? []).map((r: any) => (
                    <tr key={r.code}>
                      <td><b>{r.name}</b></td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{r.code}</td>
                      <td style={{ fontSize: 12, color: "var(--text-muted)" }}>{r.permissions?.join(" · ") || "—"}</td>
                      <td>{r.users ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
          <div className="grid grid-2">
            <Section title={t("crispdm.constraints")}>
              <Bullets items={phaseData.constraints ?? []} />
            </Section>
            <Section title={t("crispdm.inventory")}>
              <KeyValue
                items={Object.entries(phaseData.inventory ?? {}).map(([k, v]) => [
                  t(`crispdm.inv.${k}`, k),
                  String(v),
                ])}
              />
            </Section>
          </div>
        </>
      )}

      {/* Fase II */}
      {activeKey === "data-understanding" && phaseData && phaseData.counts && (
        <>
          <div className="grid grid-4" style={{ marginBottom: 22 }}>
            <StatCard label={t("crispdm.tracts")} value={phaseData.counts.tracts} subtext={`${phaseData.counts.counties ?? 0} ${t("crispdm.counties")}`} icon={<Database size={22} style={{ color: "#34d399" }} />} />
            <StatCard label={t("crispdm.indicators")} value={phaseData.counts.indicators} subtext={`${phaseData.domains?.length ?? 0} ${t("crispdm.domains")}`} icon={<Database size={22} style={{ color: "#60a5fa" }} />} />
            <StatCard label={t("crispdm.values")} value={phaseData.counts.values} subtext={`${num(phaseData.quality?.completeness)} % ${t("crispdm.completeness")}`} icon={<CheckCircle2 size={22} style={{ color: "#a78bfa" }} />} />
            <StatCard label={t("crispdm.population")} value={num(phaseData.counts.population, 0)} subtext={phaseData.dataset_nature} icon={<Target size={22} style={{ color: "#fbbf24" }} />} />
          </div>
          {phaseData.warning && (
            <div className="card" style={{ marginBottom: 22, borderColor: "var(--risk-moderate-border)" }}>
              <b style={{ color: "#fbbf24" }}>⚠ {t("crispdm.datasetWarning")}</b>
              <p style={{ color: "var(--text-muted)", margin: "8px 0 0", lineHeight: 1.55 }}>{phaseData.warning}</p>
            </div>
          )}
          <Section title={t("crispdm.profile")}>
            <div className="table-responsive" style={{ maxHeight: 420, overflowY: "auto" }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>{t("crispdm.code")}</th><th>{t("crispdm.domain")}</th><th>{t("crispdm.source")}</th>
                    <th>{t("crispdm.direction")}</th><th>n</th><th>{t("crispdm.completeness")}</th>
                    <th>min</th><th>max</th><th>media</th>
                  </tr>
                </thead>
                <tbody>
                  {(phaseData.profile ?? []).map((p: any) => (
                    <tr key={p.code}>
                      <td style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>{p.code}</td>
                      <td>{p.domain}</td>
                      <td style={{ fontSize: 12, color: "var(--text-muted)" }}>{p.source}</td>
                      <td><span className={`badge ${p.direction === "riesgo" ? "red" : "green"}`}>{p.direction}</span></td>
                      <td>{p.observations}</td>
                      <td><span className={`badge ${p.completeness >= 80 ? "green" : p.completeness > 0 ? "moderate" : "gray"}`}>{num(p.completeness)} %</span></td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(p.min)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(p.max)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(p.mean)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
          <div className="grid grid-2">
            <Section title={t("crispdm.sources")}>
              <KeyValue items={(phaseData.sources ?? []).map((s: any) => [s.source, `${s.indicators}`])} />
              <div style={{ marginTop: 14 }}>
                <KeyValue
                  items={[
                    [t("crispdm.unitOfAnalysis"), phaseData.unit_of_analysis ?? "—"],
                    ...(phaseData.years ?? []).map((y: any) => [`${t("common.year")} ${y.year}`, `${num(y.values, 0)} ${t("crispdm.values").toLowerCase()}`] as [string, string]),
                  ]}
                />
              </div>
            </Section>
            <Section title={t("crispdm.qualityChecks")}>
              <Bullets items={phaseData.quality?.checks ?? []} />
            </Section>
          </div>
        </>
      )}

      {/* Fase III */}
      {activeKey === "data-preparation" && phaseData && phaseData.coverage && (
        <>
          <div className="grid grid-4" style={{ marginBottom: 22 }}>
            <StatCard label={t("crispdm.tractsWithData")} value={phaseData.coverage.tracts_with_data} subtext={`${phaseData.coverage.tracts_total ?? 0} ${t("crispdm.total")}`} icon={<Wand2 size={22} style={{ color: "#a78bfa" }} />} />
            <StatCard label={t("crispdm.tractsComplete")} value={phaseData.coverage.tracts_complete} subtext={`${phaseData.coverage.tracts_partial ?? 0} ${t("crispdm.partial")}`} icon={<CheckCircle2 size={22} style={{ color: "#10b981" }} />} />
            <StatCard label={t("crispdm.weightsSum")} value={num(phaseData.weights_sum, 4)} subtext="Σw = 1" icon={<Brain size={22} style={{ color: "#fbbf24" }} />} />
            <StatCard label={t("crispdm.prepTime")} value={`${num(phaseData.elapsed_seconds, 4)} s`} icon={<Gauge size={22} style={{ color: "#fb923c" }} />} />
          </div>
          <div className="grid grid-2">
            <Section title={t("crispdm.etlSteps")}>
              <Bullets items={(phaseData.steps ?? []).map((s: any) => `${s.step}: ${s.detail}`)} />
            </Section>
            <Section title={t("crispdm.transformations")}>
              <Bullets items={phaseData.transformations ?? []} />
            </Section>
          </div>
          {(phaseData.sample ?? []).length > 0 && (
            <Section title={t("crispdm.sample")}>
              <div className="table-responsive">
                <table className="table">
                  <thead>
                    <tr><th>GEOID</th><th>{t("crispdm.indicators")}</th><th>{t("crispdm.meanScore")}</th><th>min</th><th>max</th></tr>
                  </thead>
                  <tbody>
                    {phaseData.sample.map((r: any) => {
                      const vals = Object.values(r.scores ?? {}) as number[];
                      const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
                      return (
                        <tr key={r.tract_id}>
                          <td style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>{r.geoid}</td>
                          <td>{vals.length}</td>
                          <td style={{ fontFamily: "var(--font-mono)" }}>{num(mean, 4)}</td>
                          <td style={{ fontFamily: "var(--font-mono)" }}>{num(vals.length ? Math.min(...vals) : null, 4)}</td>
                          <td style={{ fontFamily: "var(--font-mono)" }}>{num(vals.length ? Math.max(...vals) : null, 4)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Section>
          )}
          <Section title={t("crispdm.normalizedIndicators")}>
            <div className="table-responsive" style={{ maxHeight: 400, overflowY: "auto" }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>{t("crispdm.code")}</th><th>{t("crispdm.domain")}</th><th>{t("crispdm.inverted")}</th>
                    <th>{t("crispdm.rawWeight")}</th><th>{t("crispdm.normWeight")}</th>
                    <th>{t("crispdm.tracts")}</th><th>min</th><th>max</th><th>media</th>
                  </tr>
                </thead>
                <tbody>
                  {(phaseData.indicators ?? []).map((i: any) => (
                    <tr key={i.code}>
                      <td style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>{i.code}</td>
                      <td>{i.domain}</td>
                      <td><span className={`badge ${i.inverted ? "blue" : "gray"}`}>{i.inverted ? "1 − x̃" : "x̃"}</span></td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(i.raw_weight, 3)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(i.normalized_weight, 4)}</td>
                      <td>{i.tracts}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(i.min, 3)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(i.max, 3)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(i.mean, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </>
      )}

      {/* Fase IV */}
      {activeKey === "modeling" && phaseData && (
        <>
          <Section title={t("crispdm.technique")}>
            <p style={{ fontWeight: 600, fontSize: 15 }}>{phaseData.technique}</p>
            <p style={{ color: "var(--text-muted)", lineHeight: 1.6 }}>{phaseData.rationale}</p>
            {phaseData.formula && (
              <div style={{ background: "var(--bg-surface)", border: "1px solid var(--border-subtle)", borderRadius: 8, padding: 14, fontFamily: "var(--font-mono)", fontSize: 13.5, display: "flex", flexDirection: "column", gap: 6 }}>
                <span>{phaseData.formula.composite}</span>
                <span style={{ color: "var(--text-muted)" }}>{t("crispdm.protective")}: {phaseData.formula.protective}</span>
                <span style={{ color: "var(--text-muted)" }}>{t("crispdm.risk")}: {phaseData.formula.risk}</span>
                <span>{phaseData.formula.vulnerability}</span>
              </div>
            )}
            <p style={{ color: "var(--text-dim)", fontSize: 12.5, marginBottom: 0 }}>{phaseData.complexity}</p>
          </Section>
          <div className="grid grid-2">
            <Section title={t("crispdm.riskThresholds")}>
              <KeyValue items={(phaseData.risk_thresholds ?? []).map((r: any) => [r.level, `≥ P${r.min_percentile}`])} />
            </Section>
            <Section title={t("crispdm.assumptions")}>
              <Bullets items={phaseData.assumptions ?? []} />
            </Section>
          </div>
          <Section title={t("crispdm.weights")}>
            <div className="table-responsive" style={{ maxHeight: 380, overflowY: "auto" }}>
              <table className="table">
                <thead>
                  <tr><th>{t("crispdm.code")}</th><th>{t("common.name")}</th><th>{t("crispdm.domain")}</th><th>{t("crispdm.direction")}</th><th>{t("crispdm.rawWeight")}</th><th>{t("crispdm.normWeight")}</th></tr>
                </thead>
                <tbody>
                  {(phaseData.weights ?? []).map((w: any) => (
                    <tr key={w.code}>
                      <td style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>{w.code}</td>
                      <td>{w.name}</td>
                      <td>{w.domain}</td>
                      <td><span className={`badge ${w.direction === "riesgo" ? "red" : "green"}`}>{w.direction}</span></td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(w.raw, 3)}</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{num(w.normalized, 4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
          {phaseData.stored_indexes && (
            <Section title={`${t("crispdm.storedIndexes")} · ${phaseData.stored_indexes.total}`}>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                {["low", "moderate", "high", "critical"].map((k) => (
                  <span key={k} className={`badge ${k}`}>
                    {t(`ml.levels.${k}`)} · {phaseData.stored_indexes.by_risk?.[k] ?? 0}
                  </span>
                ))}
              </div>
              <p style={{ color: "var(--text-dim)", fontSize: 12.5, marginBottom: 0 }}>
                {(phaseData.stored_indexes.by_method ?? []).map((m: any) => `${m.method}: ${m.count}`).join(" · ")}
              </p>
            </Section>
          )}
          <MlSummary phase="modeling" />
          <Section title={t("crispdm.companionModels")}>
            <Bullets items={(phaseData.companion_models ?? []).map((m: any) => `${m.name} — ${m.detail}${m.rules != null ? ` (${m.rules})` : ""}`)} />
          </Section>
        </>
      )}

      {/* Fase V */}
      {activeKey === "evaluation" && phaseData && (
        <>
          {bench ? (
            <>
            <div className="grid grid-4" style={{ marginBottom: 10 }}>
              <StatCard label={t("crispdm.medianLatency")} value={`${num(bench.latency.median_seconds, 4)} s`} subtext={`p95 ${num(bench.latency.p95_seconds, 4)} s`} icon={<Gauge size={22} style={{ color: "#fb923c" }} />} />
              <StatCard label={t("crispdm.projected")} value={bench.latency.projected_at_target_population != null ? `${num(bench.latency.projected_at_target_population, 3)} s` : "—"} subtext={`${num(bench.latency.target_population, 0)} hab.`} icon={<Target size={22} style={{ color: "#60a5fa" }} />} />
              <StatCard label="H3" value={bench.latency.meets_h3 ? t("crispdm.met") : t("crispdm.notMet")} subtext={`< ${bench.latency.target_seconds} s`} icon={bench.latency.meets_h3 ? <CheckCircle2 size={22} style={{ color: "#10b981" }} /> : <XCircle size={22} style={{ color: "#ef4444" }} />} />
              <StatCard label={t("crispdm.stability")} value={bench.sensitivity?.risk_level_stability_pct != null ? `${num(bench.sensitivity.risk_level_stability_pct)} %` : "—"} subtext={`ρ ${num(bench.sensitivity?.spearman_mean, 4)}`} icon={<ClipboardCheck size={22} style={{ color: "#a78bfa" }} />} />
            </div>
            <p style={{ color: "var(--text-dim)", fontSize: 12.5, margin: "0 0 22px" }}>
              {t("crispdm.lastRun")}: {bench.executed_at?.replace("T", " ").replace("Z", "")} UTC · {bench.scale?.tracts} {t("crispdm.tracts")} · {bench.sensitivity?.runs} × ±{num((bench.sensitivity?.perturbation ?? 0) * 100, 0)} %
            </p>
            </>
          ) : (
            <div className="card" style={{ marginBottom: 22 }}>
              <p style={{ margin: 0, color: "var(--text-muted)" }}>{t("crispdm.benchmarkHint")}</p>
            </div>
          )}

          <Section title={t("crispdm.criteria")}>
            {phaseData.criteria ? (
              <div className="table-responsive">
                <table className="table">
                  <thead><tr><th>{t("crispdm.dimension")}</th><th>{t("crispdm.metric")}</th><th>{t("crispdm.hypothesis")}</th><th>{t("crispdm.target")}</th></tr></thead>
                  <tbody>
                    {phaseData.criteria.map((c: any) => (
                      <tr key={c.metric}>
                        <td><b>{c.dimension}</b></td>
                        <td style={{ color: "var(--text-muted)" }}>{c.metric}</td>
                        <td style={{ fontFamily: "var(--font-mono)" }}>{c.hypothesis}</td>
                        <td>{c.target}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <Bullets items={phaseData.process_review ?? []} />
            )}
          </Section>

          {phaseData.risk_distribution && Object.keys(phaseData.risk_distribution).length > 0 && (
            <Section title={t("crispdm.riskDistribution")}>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={Object.entries(phaseData.risk_distribution).map(([k, v]) => ({ name: k, count: v as number }))} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="name" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip cursor={{ fill: "rgba(255,255,255,0.04)" }} contentStyle={{ background: "var(--bg-surface)", border: "1px solid rgba(59,130,246,0.3)", borderRadius: 8 }} />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {Object.keys(phaseData.risk_distribution).map((k) => (
                      <Cell key={k} fill={RISK_COLORS[k] ?? "#64748b"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Section>
          )}

          <MlSummary phase="evaluation" />

          <MlEvaluation data={mlEval} />

          {phaseData.process_review && phaseData.criteria && (
            <Section title={t("crispdm.processReview")}>
              <Bullets items={phaseData.process_review} />
            </Section>
          )}
          {(phaseData.limitations || phaseData.note) && (
            <Section title={t("crispdm.limitations")}>
              <Bullets items={phaseData.limitations ?? [phaseData.note]} />
            </Section>
          )}
        </>
      )}

      {/* Fase VI */}
      {activeKey === "deployment" && phaseData && (
        <>
          <div className="grid grid-4" style={{ marginBottom: 22 }}>
            <StatCard label={t("crispdm.auditEvents")} value={phaseData.governance?.audit_events ?? 0} subtext={`${phaseData.governance?.users ?? 0} ${t("crispdm.users")}`} icon={<ClipboardCheck size={22} style={{ color: "#a78bfa" }} />} />
            <StatCard label={t("crispdm.reports")} value={phaseData.reports?.total ?? 0} subtext={Object.keys(phaseData.reports?.by_format ?? {}).join(" · ") || "—"} icon={<Rocket size={22} style={{ color: "#f472b6" }} />} />
            <StatCard label={t("crispdm.openAlerts")} value={phaseData.monitoring?.alerts_open ?? 0} subtext={`${phaseData.monitoring?.alert_rules ?? 0} ${t("crispdm.rules")}`} icon={<Target size={22} style={{ color: "#fb923c" }} />} />
            <StatCard label={t("crispdm.rolesPerms")} value={`${phaseData.governance?.roles ?? 0}/${phaseData.governance?.permissions ?? 0}`} icon={<CheckCircle2 size={22} style={{ color: "#10b981" }} />} />
          </div>
          <Section title={t("crispdm.services")}>
            <div className="table-responsive">
              <table className="table">
                <thead><tr><th>{t("common.name")}</th><th>{t("common.description")}</th><th>Runtime</th><th>{t("common.status")}</th></tr></thead>
                <tbody>
                  {(phaseData.services ?? []).map((s: any) => (
                    <tr key={s.name}>
                      <td><b>{s.name}</b></td>
                      <td style={{ color: "var(--text-muted)" }}>{s.detail}</td>
                      <td style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>{s.runtime}</td>
                      <td><span className={`badge ${s.status === "running" || s.status === "connected" ? "green" : "gray"}`}>{s.status}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
          <MlSummary phase="deployment" />
          <Section title={t("crispdm.recentReports")}>
            {(phaseData.reports?.recent ?? []).length === 0 ? (
              <div className="empty-state">{t("crispdm.noReports")}</div>
            ) : (
              <div className="table-responsive" style={{ maxHeight: 300, overflowY: "auto" }}>
                <table className="table">
                  <thead><tr><th>{t("crispdm.file")}</th><th>{t("crispdm.format")}</th><th>KB</th><th>{t("common.date")}</th></tr></thead>
                  <tbody>
                    {phaseData.reports.recent.map((r: any) => (
                      <tr key={r.file}>
                        <td style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>{r.file}</td>
                        <td><span className="badge blue">{r.format}</span></td>
                        <td style={{ fontFamily: "var(--font-mono)" }}>{num(r.size_kb, 1)}</td>
                        <td style={{ fontFamily: "var(--font-mono)", fontSize: 11.5 }}>{r.modified?.replace("T", " ").replace("Z", "")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>
          <div className="grid grid-2">
            <Section title={t("crispdm.feedbackLoop")}>
              <Bullets items={phaseData.feedback_loop ?? []} />
            </Section>
            <Section title={t("crispdm.recentAudit")}>
              {(phaseData.recent_audit ?? []).length === 0 ? (
                <div className="empty-state">{t("crispdm.noAudit")}</div>
              ) : (
                <div className="table-responsive" style={{ maxHeight: 300, overflowY: "auto" }}>
                  <table className="table">
                    <thead><tr><th>{t("common.date")}</th><th>{t("crispdm.user")}</th><th>{t("crispdm.action")}</th><th>{t("common.description")}</th></tr></thead>
                    <tbody>
                      {(phaseData.recent_audit ?? []).map((a: any) => (
                        <tr key={a.id}>
                          <td style={{ fontFamily: "var(--font-mono)", fontSize: 11.5 }}>{a.created_at?.replace("T", " ").replace("Z", "")}</td>
                          <td>{a.username || "—"}</td>
                          <td><span className="badge blue">{a.action}</span></td>
                          <td style={{ color: "var(--text-muted)", fontSize: 12 }}>{a.detail || a.resource || "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Section>
          </div>
        </>
      )}
    </div>
  );
}
