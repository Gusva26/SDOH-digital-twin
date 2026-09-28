import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Activity,
  AlertTriangle,
  Building2,
  Cpu,
  Hospital as HospitalIcon,
  MapPin,
  Scale,
  Sparkles,
  Layers,
  Users,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell as RechartsCell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import api from "../api/client";
import StatCard from "../components/StatCard";
import Twin3D, { TwinCell, TwinHospital } from "../components/Twin3D";
import type { Alert, EquityIndex, Hospital } from "../types";

interface CellInfo {
  tract_geoid?: string;
  value: number;
  risk_level?: string;
  percentile?: number;
  population?: number;
}

const RISK_COLORS: Record<string, string> = {
  low: "#10b981",
  moderate: "#f59e0b",
  high: "#f97316",
  critical: "#ef4444",
};

const YEAR = 2023;

export default function Dashboard() {
  const { t } = useTranslation();
  const [stats, setStats] = useState<any>(null);
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [hospitalId, setHospitalId] = useState<number | null>(null);
  const [equity, setEquity] = useState<EquityIndex[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [colorBy, setColorBy] = useState<"value" | "risk" | "ml">("risk");
  const [mlPreds, setMlPreds] = useState<Record<string, any>>({});
  const [mlModel, setMlModel] = useState<{ model: string; target: string } | null>(null);
  const [selected, setSelected] = useState<TwinCell | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<CellInfo | null>(null);
  const [loadingTwin, setLoadingTwin] = useState(false);

  useEffect(() => {
    api.get("/geo/health-system/stats").then((r) => setStats(r.data)).catch(() => {});
    api.get("/sdoh/alerts?limit=6").then((r) => setAlerts(r.data)).catch(() => {});
    // Predicciones del modelo ML desplegado (409 si aún no se entrena: el modo queda deshabilitado).
    api
      .get("/ml/predictions")
      .then((r) => {
        setMlPreds(Object.fromEntries(r.data.items.map((i: any) => [i.geoid, i])));
        setMlModel({ model: r.data.model, target: r.data.target?.name });
      })
      .catch(() => setMlModel(null));
    api
      .get<Hospital[]>("/hospitals")
      .then((r) => {
        setHospitals(r.data);
        if (r.data.length) setHospitalId(r.data[0].id);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (hospitalId == null) {
      setEquity([]);
      return;
    }
    let cancelled = false;
    setLoadingTwin(true);
    api
      .get<EquityIndex[]>("/sdoh/equity", { params: { year: YEAR, hospital_id: hospitalId } })
      .then((r) => {
        if (cancelled) return;
        setEquity(r.data);
      })
      .catch(() => {
        if (!cancelled) setEquity([]);
      })
      .finally(() => {
        if (!cancelled) setLoadingTwin(false);
      });
    return () => {
      cancelled = true;
    };
  }, [hospitalId]);

  const cells: TwinCell[] = useMemo(() => {
    return equity.map((e) => ({
      id: String(e.tract_id),
      x: 0,
      z: 0,
      label: e.tract_geoid ? `Tract ${e.tract_geoid.slice(-4)}` : `Tract ${e.tract_id}`,
      name: e.tract_geoid || undefined,
      value: e.value || 0,
      percentile: e.percentile ?? undefined,
      population: e.population,
      risk:
        colorBy === "ml"
          ? mlPreds[e.tract_geoid ?? ""]?.risk_level || "low"
          : e.risk_level || "low",
    }));
  }, [equity, colorBy, mlPreds]);

  const hospital: TwinHospital | null = useMemo(() => {
    const found = hospitals.find((h) => h.id === hospitalId);
    return found ? { id: found.id, name: found.name, city: found.city, state: found.state } : null;
  }, [hospitals, hospitalId]);

  const switchHospital = useCallback((id: number) => {
    setHospitalId(id);
    setSelected(null);
    setSelectedDetail(null);
  }, []);

  const riskCounts = useMemo(() => {
    const counts: Record<string, number> = { low: 0, moderate: 0, high: 0, critical: 0 };
    cells.forEach((c) => {
      counts[c.risk] = (counts[c.risk] || 0) + 1;
    });
    return [
      { name: t("common.low"), key: "low", count: counts.low, fill: RISK_COLORS.low },
      { name: t("common.moderate"), key: "moderate", count: counts.moderate, fill: RISK_COLORS.moderate },
      { name: t("common.high"), key: "high", count: counts.high, fill: RISK_COLORS.high },
      { name: t("common.critical"), key: "critical", count: counts.critical, fill: RISK_COLORS.critical },
    ];
  }, [cells, t]);

  const handleSelect = (cell: TwinCell) => {
    setSelected(cell);
    const eq = equity.find((e) => String(e.tract_id) === cell.id);
    if (eq) {
      setSelectedDetail({
        tract_geoid: eq.tract_geoid,
        value: eq.value,
        risk_level: eq.risk_level,
        percentile: eq.percentile,
        population: (eq as EquityIndex & { population?: number }).population,
      });
    }
  };

  return (
    <div>
      <div className="topbar">
        <div className="page-title-group">
          <h2 className="page-title">{t("dashboard.title")}</h2>
          <p className="page-subtitle">{t("dashboard.subtitle")}</p>
        </div>
      </div>

      <div className="grid grid-4" style={{ marginBottom: 22 }}>
        <StatCard
          label={t("dashboard.registeredHospitals")}
          value={stats?.hospitals ?? "—"}
          icon={<Building2 size={22} />}
          subtext={t("dashboard.hospitalsSubtext")}
        />
        <StatCard
          label={t("dashboard.catchmentTracts")}
          value={cells.length || "—"}
          icon={<MapPin size={22} />}
          subtext={hospital?.name ?? t("dashboard.tractsSubtext")}
        />
        <StatCard
          label={t("dashboard.catchmentAreas")}
          value={stats?.catchments ?? "—"}
          icon={<Activity size={22} />}
          subtext={t("dashboard.catchmentSubtext")}
        />
        <StatCard
          label={t("dashboard.openAlerts")}
          value={alerts.filter((a) => a.status === "open").length}
          icon={<AlertTriangle size={22} style={{ color: "#f87171" }} />}
          subtext={t("dashboard.alertsSubtext")}
        />
      </div>

      <div className="twin-card card">
        <div className="twin-card-head">
          <div className="twin-card-headings">
            <div className="twin-card-eyebrow">
              <span className="status-dot" />
              {t("dashboard.twinLive")}
            </div>
            <h3 className="twin-card-title">{t("dashboard.twinTitle")}</h3>
          </div>

          <div className="twin-card-actions">
            <div className="twin-segmented" role="tablist" aria-label={t("dashboard.selectHospital")}>
              {hospitals.map((h) => {
                const active = hospitalId === h.id;
                return (
                  <button
                    key={h.id}
                    type="button"
                    role="tab"
                    aria-selected={active}
                    className={`twin-segment ${active ? "active" : ""}`}
                    onClick={() => switchHospital(h.id)}
                  >
                    <HospitalIcon size={13} />
                    <span>{h.name}</span>
                  </button>
                );
              })}
              {hospitals.length === 0 && (
                <div className="twin-segment-placeholder">{t("common.loading")}</div>
              )}
            </div>

            <div className="twin-segmented">
              <button
                type="button"
                className={`twin-segment ${colorBy === "risk" ? "active" : ""}`}
                onClick={() => setColorBy("risk")}
              >
                <Sparkles size={13} />
                <span>{t("dashboard.byRisk")}</span>
              </button>
              <button
                type="button"
                className={`twin-segment ${colorBy === "value" ? "active" : ""}`}
                onClick={() => setColorBy("value")}
              >
                <Layers size={13} />
                <span>{t("dashboard.byEquity")}</span>
              </button>
              <button
                type="button"
                className={`twin-segment ${colorBy === "ml" ? "active" : ""}`}
                onClick={() => setColorBy("ml")}
                disabled={!mlModel}
                title={mlModel ? `${mlModel.model} · ${mlModel.target}` : t("dashboard.mlNotTrained")}
              >
                <Cpu size={13} />
                <span>{t("dashboard.byMl")}</span>
              </button>
            </div>
          </div>
        </div>

        <Twin3D
          cells={cells}
          colorBy={colorBy === "value" ? "value" : "risk"}
          hudLabel={colorBy === "ml" && mlModel ? `${t("dashboard.coloredByMl")} · ${mlModel.target}` : undefined}
          onSelect={handleSelect}
          hospital={hospital}
          loading={loadingTwin}
        />
      </div>

      {selectedDetail && selected && (
        <div
          className="card"
          style={{
            marginBottom: 22,
            borderColor: "var(--border-card)",
            background: "var(--bg-surface-elevated)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: "50%",
                  background: RISK_COLORS[selectedDetail.risk_level || "low"],
                  boxShadow: `0 0 10px ${RISK_COLORS[selectedDetail.risk_level || "low"]}`,
                }}
              />
              <h3 style={{ margin: 0, fontSize: 16 }}>
                {t("dashboard.tractDetail")}: <span style={{ color: "#60a5fa" }}>{selectedDetail.tract_geoid || selected.id}</span>
              </h3>
            </div>
            <button
              className="btn btn-sm secondary"
              onClick={() => { setSelected(null); setSelectedDetail(null); }}
            >
              {t("dashboard.closeInspector")}
            </button>
          </div>

          <div className="grid grid-3">
            <StatCard
              label={t("dashboard.equityIndex")}
              value={selectedDetail.value != null ? selectedDetail.value.toFixed(3) : "—"}
              icon={<Scale size={20} />}
              subtext={t("dashboard.equitySubtext")}
            />
            <StatCard
              label={t("dashboard.percentile")}
              value={selectedDetail.percentile != null ? `${selectedDetail.percentile.toFixed(1)}%` : "—"}
              icon={<Activity size={20} />}
              subtext={t("dashboard.percentileSubtext")}
            />
            <StatCard
              label={t("dashboard.population")}
              value={selectedDetail.population != null ? selectedDetail.population.toLocaleString() : "—"}
              icon={<Users size={20} />}
              subtext={t("dashboard.populationSubtext")}
            />
            <div className="card">
              <div className="stat-label">{t("dashboard.riskClassification")}</div>
              <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 10 }}>
                <span className={`badge ${selectedDetail.risk_level}`}>{selectedDetail.risk_level}</span>
                <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{t("dashboard.riskPriority")}</span>
              </div>
            </div>
            {mlModel && selectedDetail.tract_geoid && mlPreds[selectedDetail.tract_geoid] && (() => {
              const p = mlPreds[selectedDetail.tract_geoid!];
              return (
                <div className="card">
                  <div className="stat-label">{t("dashboard.mlPrediction")} · {mlModel.target}</div>
                  <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                    <span className={`badge ${p.risk_level}`}>{p.risk_level}</span>
                    <span style={{ fontSize: 12, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                      {(p.confidence * 100).toFixed(1)} %
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 8 }}>
                    {t("dashboard.mlObserved")}: {p.observed_value ?? "—"} %{p.observed_level ? ` (${p.observed_level})` : ""} · {mlModel.model}
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      )}

      <div className="grid grid-2">
        <div className="chart-card">
          <h3>
            <span>{t("dashboard.riskDistribution")}</span>
            <span className="chart-h3-meta">
              {hospital ? hospital.name : t("dashboard.allHospitals")} · {YEAR}
            </span>
          </h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={riskCounts} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 12 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 12 }} />
              <Tooltip
                cursor={{ fill: "rgba(255,255,255,0.04)" }}
                contentStyle={{
                  background: "var(--bg-surface)",
                  border: "1px solid rgba(59,130,246,0.3)",
                  borderRadius: "8px",
                  boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
                }}
              />
              <Bar dataKey="count" radius={[6, 6, 0, 0]} name={t("dashboard.censusTractsLabel")}>
                {riskCounts.map((entry) => (
                  <RechartsCell key={entry.key} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>
            <span>{t("dashboard.recentAlerts")}</span>
            <span className="badge red" style={{ fontSize: 11 }}>
              {alerts.length} {t("dashboard.notifications")}
            </span>
          </h3>
          {alerts.length === 0 ? (
            <div className="empty-state">{t("dashboard.noAlerts")}</div>
          ) : (
            <div className="table-responsive">
              <table className="table">
                <thead>
                  <tr>
                    <th>{t("dashboard.sdohIndicator")}</th>
                    <th>{t("dashboard.severity")}</th>
                    <th>{t("dashboard.value")}</th>
                  </tr>
                </thead>
                <tbody>
                  {alerts.map((a) => (
                    <tr key={a.id}>
                      <td>
                        <div style={{ fontWeight: 600, color: "var(--text-main)" }}>
                          {a.indicator_name || a.indicator_code}
                        </div>
                        <div style={{ fontSize: 11, color: "var(--text-dim)" }}>{a.message}</div>
                      </td>
                      <td><span className={`badge ${a.severity}`}>{a.severity}</span></td>
                      <td style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{a.observed_value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
