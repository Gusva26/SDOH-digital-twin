import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { Database, Layers, Sparkles, Map as MapIcon, BarChart2 } from "lucide-react";
import api from "../api/client";
import StatCard from "../components/StatCard";
import SatelliteMap from "../components/SatelliteMap";
import type { Hospital, Catchment, EquityIndex } from "../types";

interface ValueRow {
  catalog_code: string;
  catalog_name: string;
  domain?: string;
  value: number;
  year: number;
}

const PIE_COLORS = ["#3b82f6", "#06b6d4", "#8b5cf6", "#10b981", "#f59e0b", "#ec4899", "#6366f1"];

export default function SdohMap() {
  const { t } = useTranslation();
  const [catalog, setCatalog] = useState<any[]>([]);
  const [values, setValues] = useState<ValueRow[]>([]);
  const [filterDomain, setFilterDomain] = useState("");
  const [searchQuery, setSearchQuery] = useState("");

  // Spatial & Hospital state
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [selectedHospitalId, setSelectedHospitalId] = useState<number | null>(null);
  const [catchments, setCatchments] = useState<Catchment[]>([]);
  const [equityList, setEquityList] = useState<EquityIndex[]>([]);
  const [activeTab, setActiveTab] = useState<"satellite" | "analytics">("satellite");

  useEffect(() => {
    api.get("/sdoh/catalog").then((r) => setCatalog(r.data)).catch(() => {});
    api.get("/sdoh/values?year=2023&limit=2000").then((r) => setValues(r.data)).catch(() => {});

    // Fetch spatial GIS data
    api.get("/hospitals").then((r) => {
      setHospitals(r.data);
      if (r.data.length > 0 && selectedHospitalId === null) {
        setSelectedHospitalId(r.data[0].id);
      }
    }).catch(() => {});

    api.get("/geo/catchments").then((r) => setCatchments(r.data)).catch(() => {});
    api.get("/sdoh/equity?year=2023").then((r) => setEquityList(r.data)).catch(() => {});
  }, []);

  const domains = useMemo(() => {
    const map = new Map<string, number>();
    catalog.forEach((c) => c.domain && map.set(c.domain, (map.get(c.domain) || 0) + 1));
    return Array.from(map.entries()).map(([name, count]) => ({ name, count, value: count }));
  }, [catalog]);

  const filtered = useMemo(() => {
    let list = values;
    if (filterDomain) {
      const codes = new Set(catalog.filter((c) => c.domain === filterDomain).map((c) => c.code));
      list = list.filter((v) => codes.has(v.catalog_code));
    }
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      list = list.filter((v) => v.catalog_name.toLowerCase().includes(q) || v.catalog_code.toLowerCase().includes(q));
    }
    return list;
  }, [values, filterDomain, searchQuery, catalog]);

  const avgByIndicator = useMemo(() => {
    const map = new Map<string, { sum: number; n: number; name: string; domain?: string }>();
    filtered.forEach((v) => {
      const cur = map.get(v.catalog_code) || { sum: 0, n: 0, name: v.catalog_name, domain: v.domain };
      cur.sum += v.value;
      cur.n += 1;
      map.set(v.catalog_code, cur);
    });
    return Array.from(map.entries())
      .map(([code, d]) => ({ code, name: d.name, domain: d.domain, avg: +(d.sum / d.n).toFixed(2), records: d.n }))
      .sort((a, b) => b.avg - a.avg);
  }, [filtered]);

  return (
    <div>
      <div className="topbar">
        <div className="page-title-group">
          <h2 className="page-title">Mapa Satelital & Determinantes SDOH</h2>
          <p className="page-subtitle">Consola de teledetección espacial y análisis geoespacial de hospitales</p>
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          {/* Tab Switcher */}
          <div style={{ display: "flex", background: "rgba(15, 23, 42, 0.8)", padding: 4, borderRadius: "var(--radius-sm)", border: "1px solid var(--border-subtle)" }}>
            <button
              className={`btn ${activeTab === "satellite" ? "" : "secondary"}`}
              style={{
                padding: "6px 14px",
                fontSize: 12.5,
                background: activeTab === "satellite" ? "linear-gradient(135deg, #0284c7 0%, #06b6d4 100%)" : "transparent",
                border: "none",
                display: "inline-flex",
                alignItems: "center",
                gap: 6
              }}
              onClick={() => setActiveTab("satellite")}
            >
              <MapIcon size={15} />
              Consola Satelital GIS
            </button>
            <button
              className={`btn ${activeTab === "analytics" ? "" : "secondary"}`}
              style={{
                padding: "6px 14px",
                fontSize: 12.5,
                background: activeTab === "analytics" ? "var(--bg-surface-elevated)" : "transparent",
                border: "none",
                display: "inline-flex",
                alignItems: "center",
                gap: 6
              }}
              onClick={() => setActiveTab("analytics")}
            >
              <BarChart2 size={15} />
              Catálogo de Indicadores
            </button>
          </div>
        </div>
      </div>

      {/* Main Satellite Console */}
      <SatelliteMap
        hospitals={hospitals}
        selectedHospitalId={selectedHospitalId}
        onSelectHospital={setSelectedHospitalId}
        catchments={catchments}
        equityList={equityList}
      />

      {/* Analytics Section (visible or highlighted when tab is analytics) */}
      <div style={{ marginTop: 28, paddingTop: 20, borderTop: "1px solid var(--border-subtle)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 16, display: "flex", alignItems: "center", gap: 8 }}>
              <Layers size={18} style={{ color: "#38bdf8" }} />
              {t("sdohMap.title")} & Distribución por Dominio
            </h3>
            <p style={{ margin: "4px 0 0", fontSize: 12, color: "var(--text-dim)" }}>
              {t("sdohMap.subtitle")}
            </p>
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <div style={{ position: "relative", width: 200 }}>
              <input
                className="input"
                style={{ padding: "6px 10px", fontSize: 12 }}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder={`${t("common.search")}…`}
              />
            </div>
            <div style={{ width: 190 }}>
              <select
                className="select"
                style={{ padding: "6px 10px", fontSize: 12 }}
                value={filterDomain}
                onChange={(e) => setFilterDomain(e.target.value)}
              >
                <option value="">{t("sdohMap.allDomains")} ({catalog.length})</option>
                {domains.map((d) => (
                  <option key={d.name} value={d.name}>{d.name} ({d.count})</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className="grid grid-3" style={{ marginBottom: 20 }}>
          <StatCard label="Total Indicadores" value={catalog.length} icon={<Layers size={22} />} subtext="CDC PLACES & ACS" />
          <StatCard label={t("sdohMap.domain") + "s SDOH"} value={domains.length} icon={<Sparkles size={22} />} subtext="Determinantes sociales" />
          <StatCard label="Registros" value={filtered.length} icon={<Database size={22} />} subtext="Observaciones (2022)" />
        </div>

        <div className="grid grid-3" style={{ marginBottom: 20 }}>
          <div className="chart-card">
            <h3>{t("sdohMap.pieChart")}</h3>
            <ResponsiveContainer width="100%" height={290}>
              <PieChart>
                <Pie data={domains} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={55} outerRadius={95} paddingAngle={4}
                  label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}>
                  {domains.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ background: "var(--bg-surface)", border: "1px solid rgba(59,130,246,0.3)", borderRadius: "8px" }} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card" style={{ gridColumn: "span 2" }}>
            <h3>
              <span>{t("sdohMap.avgByIndicator")} {filterDomain ? `· ${filterDomain}` : ""}</span>
              <span className="badge blue">{avgByIndicator.length}</span>
            </h3>
            {avgByIndicator.length === 0 ? (
              <div className="empty-state">{t("sdohMap.noData")}</div>
            ) : (
              <div className="table-responsive" style={{ maxHeight: 310, overflowY: "auto" }}>
                <table className="table">
                  <thead>
                    <tr>
                      <th>{t("common.name")}</th>
                      <th style={{ width: 140 }}>{t("sdohMap.average")}</th>
                      <th style={{ width: 100 }}>Muestras</th>
                    </tr>
                  </thead>
                  <tbody>
                    {avgByIndicator.map((r) => (
                      <tr key={r.code}>
                        <td>
                          <div style={{ fontWeight: 600, color: "var(--text-main)" }}>{r.name}</div>
                          <div style={{ fontSize: 11, color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>{r.code}</div>
                        </td>
                        <td><span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{r.avg}</span></td>
                        <td style={{ color: "var(--text-dim)", fontSize: 12 }}>{r.records} tracts</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
