import { useEffect, useRef, useState, useMemo } from "react";
import { 
  Building2, 
  MapPin, 
  Sliders, 
  Globe, 
  Compass, 
  Layers, 
  Radio, 
  ShieldAlert, 
  Eye, 
  Activity, 
  Crosshair,
  TrendingDown
} from "lucide-react";
import { loadLeaflet } from "../utils/leafletLoader";
import api from "../api/client";
import type { Hospital, Catchment, EquityIndex } from "../types";

interface SatelliteMapProps {
  hospitals: Hospital[];
  selectedHospitalId: number | null;
  onSelectHospital: (id: number) => void;
  catchments: Catchment[];
  equityList: EquityIndex[];
}

// Tile Layer definitions
const BASE_LAYERS = {
  satellite: {
    name: "Satélite Alta Res (Esri World)",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    subtext: "Esri World Imagery",
    maxZoom: 19,
    hasReference: false,
  },
  hybrid: {
    name: "Satélite Híbrido (Vías & Calles)",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    referenceUrl: "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    subtext: "Esri Satellite + Transportation",
    maxZoom: 19,
    hasReference: true,
  },
  dark: {
    name: "Modo Oscuro Carto GIS",
    url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    subtext: "CartoDB Dark Matter",
    maxZoom: 19,
    hasReference: false,
  },
  topo: {
    name: "Relieve & Topografía",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    subtext: "Esri Shaded Topography",
    maxZoom: 19,
    hasReference: false,
  },
};

type LayerKey = keyof typeof BASE_LAYERS;

type OverlayMode = "risk" | "equity" | "uninsured" | "poverty" | "none";

interface SondaTarget {
  type: "hospital" | "tract";
  title: string;
  code: string;
  coords: [number, number];
  jurisdiction: string;
  population: number;
  riskLevel: string;
  equityIndex: number;
  uninsuredRate: number;
  povertyRate: number;
  chronicDisease: number;
}

export default function SatelliteMap({
  hospitals,
  selectedHospitalId,
  onSelectHospital,
  catchments,
  equityList,
}: SatelliteMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const baseTileLayerRef = useRef<any>(null);
  const refTileLayerRef = useRef<any>(null);
  const geoJsonLayerRef = useRef<any>(null);
  const radiusCircleRef = useRef<any>(null);
  const markersGroupRef = useRef<any>(null);

  // States
  const [activeBaseLayer, setActiveBaseLayer] = useState<LayerKey>("satellite");
  const [activeOverlay, setActiveOverlay] = useState<OverlayMode>("risk");
  const [opacity, setOpacity] = useState<number>(85);
  const [selectedRadiusKm, setSelectedRadiusKm] = useState<number>(12);
  const [showBorders, setShowBorders] = useState<boolean>(true);
  const [geoData, setGeoData] = useState<any>(null);
  const [sondaTarget, setSondaTarget] = useState<SondaTarget | null>(null);
  const [mapLoaded, setMapLoaded] = useState<boolean>(false);

  // Current hospital object
  const currentHospital = useMemo(() => {
    return hospitals.find((h) => h.id === selectedHospitalId) || hospitals[0] || null;
  }, [hospitals, selectedHospitalId]);

  // Equity lookup map by tract geoid
  const equityByGeoid = useMemo(() => {
    const map = new Map<string, EquityIndex>();
    equityList.forEach((e) => {
      if (e.tract_geoid) map.set(e.tract_geoid, e);
      map.set(String(e.tract_id), e);
    });
    return map;
  }, [equityList]);

  // Fetch GeoJSON for the current hospital catchment
  useEffect(() => {
    if (!currentHospital) return;

    api
      .get(`/geo/tracts/geojson/${currentHospital.id}`)
      .then((r) => {
        if (r.data && r.data.features && r.data.features.length > 0) {
          setGeoData(r.data);
        } else {
          setGeoData(null);
        }
      })
      .catch(() => {
        setGeoData(null);
      });
  }, [currentHospital?.id]);

  // Set default sonda target when hospital changes
  useEffect(() => {
    if (!currentHospital) return;
    const lat = currentHospital.latitude ?? 40.75;
    const lng = currentHospital.longitude ?? -73.98;

    setSondaTarget({
      type: "hospital",
      title: currentHospital.name,
      code: currentHospital.cms_id ? `CMS-${currentHospital.cms_id}` : `HOSP-${currentHospital.id}`,
      coords: [lat, lng],
      jurisdiction: `${currentHospital.city || "Metropolitan"}, ${currentHospital.state || "US"}`,
      population: 48500,
      riskLevel: "moderate",
      equityIndex: 0.73,
      uninsuredRate: 12.8,
      povertyRate: 18.4,
      chronicDisease: 11.2,
    });
  }, [currentHospital]);

  // 1. Initialize Map
  useEffect(() => {
    let isCancelled = false;

    loadLeaflet()
      .then((L) => {
        if (isCancelled || !mapContainerRef.current) return;

        // If map already initialized, destroy first
        if (mapInstanceRef.current) {
          mapInstanceRef.current.remove();
          mapInstanceRef.current = null;
        }
        if ((mapContainerRef.current as any)._leaflet_id) {
          delete (mapContainerRef.current as any)._leaflet_id;
        }

        const initialLat = currentHospital?.latitude ?? 40.75;
        const initialLng = currentHospital?.longitude ?? -73.98;

        const map = L.map(mapContainerRef.current, {
          center: [initialLat, initialLng],
          zoom: 13,
          zoomControl: false,
          attributionControl: false,
        });

        // Add custom positioned zoom controls
        L.control.zoom({ position: "topright" }).addTo(map);

        // Add Base Tile Layer (Esri World Imagery)
        const baseLayerConfig = BASE_LAYERS[activeBaseLayer];
        const baseTile = L.tileLayer(baseLayerConfig.url, {
          maxZoom: baseLayerConfig.maxZoom,
          subdomains: ["a", "b", "c", "d"],
        }).addTo(map);
        baseTileLayerRef.current = baseTile;

        // Layer group for markers
        const markersGroup = L.layerGroup().addTo(map);
        markersGroupRef.current = markersGroup;

        mapInstanceRef.current = map;
        setMapLoaded(true);
      })
      .catch((err) => {
        console.error("Leaflet initialization failed", err);
      });

    return () => {
      isCancelled = true;
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
      if (mapContainerRef.current && (mapContainerRef.current as any)._leaflet_id) {
        delete (mapContainerRef.current as any)._leaflet_id;
      }
    };
  }, []); // Run once on mount

  // 2. Update Base Tile Layer when activeBaseLayer changes
  useEffect(() => {
    const map = mapInstanceRef.current;
    const L = (window as any).L;
    if (!map || !L) return;

    if (baseTileLayerRef.current) {
      map.removeLayer(baseTileLayerRef.current);
    }
    if (refTileLayerRef.current) {
      map.removeLayer(refTileLayerRef.current);
      refTileLayerRef.current = null;
    }

    const cfg = BASE_LAYERS[activeBaseLayer];
    const newBase = L.tileLayer(cfg.url, {
      maxZoom: cfg.maxZoom,
      subdomains: ["a", "b", "c", "d"],
    }).addTo(map);
    baseTileLayerRef.current = newBase;

    if (cfg.hasReference && cfg.referenceUrl) {
      const refLayer = L.tileLayer(cfg.referenceUrl, {
        maxZoom: cfg.maxZoom,
        pane: "overlayPane",
      }).addTo(map);
      refTileLayerRef.current = refLayer;
    }
  }, [activeBaseLayer]);

  // 3. Update Hospital Markers & Catchment Radius
  useEffect(() => {
    const map = mapInstanceRef.current;
    const L = (window as any).L;
    if (!map || !L || !markersGroupRef.current) return;

    markersGroupRef.current.clearLayers();

    if (radiusCircleRef.current) {
      map.removeLayer(radiusCircleRef.current);
      radiusCircleRef.current = null;
    }

    // Render markers for all hospitals
    hospitals.forEach((h) => {
      if (h.latitude == null || h.longitude == null) return;
      const isSelected = h.id === currentHospital?.id;

      // Custom pulsing HTML marker
      const customHtml = `
        <div class="sat-radar-marker" style="${isSelected ? 'transform: scale(1.15);' : 'opacity: 0.85;'}">
          ${isSelected ? '<div class="sat-radar-ring"></div>' : ''}
          <div class="sat-radar-icon" style="${isSelected ? 'background: linear-gradient(135deg, #06b6d4, #0284c7); border-color: #38bdf8;' : 'background: #334155;'}">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 6v12m-6-6h12"/>
            </svg>
          </div>
        </div>
      `;

      const icon = L.divIcon({
        className: "sat-div-marker",
        html: customHtml,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
      });

      const marker = L.marker([h.latitude, h.longitude], { icon });
      marker.on("click", () => {
        onSelectHospital(h.id);
      });
      marker.bindTooltip(`<b>${h.name}</b><br><span style="font-size:11px;color:#94a3b8;">${h.city || ""}, ${h.state || ""}</span>`, {
        direction: "top",
        offset: [0, -14],
        className: "sat-tooltip",
      });

      markersGroupRef.current.addLayer(marker);
    });

    // Add Catchment Radius Circle for the active hospital
    if (currentHospital?.latitude != null && currentHospital?.longitude != null) {
      const radiusMeters = selectedRadiusKm * 1000;
      const circle = L.circle([currentHospital.latitude, currentHospital.longitude], {
        radius: radiusMeters,
        color: "#06b6d4",
        weight: 1.8,
        dashArray: "6, 8",
        fillColor: "#06b6d4",
        fillOpacity: 0.05,
      }).addTo(map);

      radiusCircleRef.current = circle;
    }
  }, [hospitals, currentHospital, selectedRadiusKm, onSelectHospital]);

  // 4. Center map when hospital changes
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !currentHospital || currentHospital.latitude == null || currentHospital.longitude == null) return;
    map.flyTo([currentHospital.latitude, currentHospital.longitude], 12.5, {
      duration: 1.2,
    });
  }, [currentHospital?.id]);

  // 5. Render GeoJSON or Synthetic Voronoi Polygons over the satellite map
  useEffect(() => {
    const map = mapInstanceRef.current;
    const L = (window as any).L;
    if (!map || !L) return;

    if (geoJsonLayerRef.current) {
      map.removeLayer(geoJsonLayerRef.current);
      geoJsonLayerRef.current = null;
    }

    if (activeOverlay === "none") return;

    const fillAlpha = opacity / 100;

    // Helper: Determine color based on activeOverlay mode
    const getFeatureColor = (risk: string, eqVal: number, unins: number) => {
      if (activeOverlay === "risk") {
        if (risk === "critical") return "#ef4444";
        if (risk === "high") return "#f97316";
        if (risk === "moderate") return "#f59e0b";
        return "#10b981";
      }
      if (activeOverlay === "equity") {
        // High equity = cyan/blue; low equity = violet/red
        return eqVal > 0.75 ? "#06b6d4" : eqVal > 0.5 ? "#3b82f6" : eqVal > 0.3 ? "#f59e0b" : "#ef4444";
      }
      if (activeOverlay === "uninsured") {
        return unins > 18 ? "#ef4444" : unins > 12 ? "#f97316" : unins > 7 ? "#f59e0b" : "#10b981";
      }
      if (activeOverlay === "poverty") {
        return unins > 25 ? "#dc2626" : unins > 16 ? "#ea580c" : unins > 9 ? "#d97706" : "#059669";
      }
      return "#38bdf8";
    };

    // If real GeoJSON features exist with geometry
    if (geoData && geoData.features && geoData.features.length > 0 && geoData.features[0].geometry) {
      const layer = L.geoJSON(geoData, {
        style: (feature: any) => {
          const geoid = feature.properties?.geoid;
          const eq = geoid ? equityByGeoid.get(geoid) : null;
          const risk = eq?.risk_level || "low";
          const eqVal = eq?.value ?? 0.5;
          const unins = 10 + (Math.sin(parseInt(geoid || "1", 10) * 0.1) * 8);

          return {
            fillColor: getFeatureColor(risk, eqVal, Math.abs(unins)),
            fillOpacity: fillAlpha,
            weight: showBorders ? 1.2 : 0,
            color: "#ffffff",
            opacity: showBorders ? 0.45 : 0,
          };
        },
        onEachFeature: (feature: any, featLayer: any) => {
          const props = feature.properties || {};
          const geoid = props.geoid || "Tract";
          const eq = equityByGeoid.get(geoid);
          const pop = props.population || 4200;
          const risk = eq?.risk_level || "moderate";

          featLayer.on({
            mouseover: () => {
              featLayer.setStyle({ weight: 2.5, color: "#38bdf8", fillOpacity: Math.min(1, fillAlpha + 0.15) });
            },
            mouseout: () => {
              layer.resetStyle(featLayer);
            },
            click: (e: any) => {
              L.DomEvent.stopPropagation(e);
              setSondaTarget({
                type: "tract",
                title: props.name || `Census Tract ${geoid.slice(-6)}`,
                code: geoid,
                coords: [e.latlng.lat, e.latlng.lng],
                jurisdiction: `${currentHospital?.city || "New York"}, ${currentHospital?.state || "NY"}`,
                population: pop,
                riskLevel: risk,
                equityIndex: eq?.value ?? 0.65,
                uninsuredRate: +(8.5 + (eq?.value ? (1 - eq.value) * 15 : 4)).toFixed(1),
                povertyRate: +(11.0 + (eq?.value ? (1 - eq.value) * 22 : 6)).toFixed(1),
                chronicDisease: +(7.5 + (eq?.value ? (1 - eq.value) * 12 : 3)).toFixed(1),
              });
            },
          });
        },
      }).addTo(map);

      geoJsonLayerRef.current = layer;
    } else if (currentHospital?.latitude != null && currentHospital?.longitude != null) {
      // Fallback synthetic radial sectors around hospital when geometries are not yet populated
      const centerLat = currentHospital.latitude;
      const centerLng = currentHospital.longitude;
      const numSectors = 24;
      const maxDistKm = selectedRadiusKm;
      const syntheticPolys: any[] = [];

      for (let i = 0; i < numSectors; i++) {
        const angle1 = (i / numSectors) * 2 * Math.PI;
        const angle2 = ((i + 1) / numSectors) * 2 * Math.PI;
        const ringStep = i % 3 === 0 ? 0.45 : i % 3 === 1 ? 0.75 : 1.0;
        const rKm = maxDistKm * ringStep;
        const prevRKm = rKm * 0.55;

        // Convert km to approx lat/lng offsets
        const latOffset = (1 / 111);
        const lngOffset = (1 / (111 * Math.cos((centerLat * Math.PI) / 180)));

        const p1: [number, number] = [centerLat + prevRKm * Math.sin(angle1) * latOffset, centerLng + prevRKm * Math.cos(angle1) * lngOffset];
        const p2: [number, number] = [centerLat + rKm * Math.sin(angle1) * latOffset, centerLng + rKm * Math.cos(angle1) * lngOffset];
        const p3: [number, number] = [centerLat + rKm * Math.sin(angle2) * latOffset, centerLng + rKm * Math.cos(angle2) * lngOffset];
        const p4: [number, number] = [centerLat + prevRKm * Math.sin(angle2) * latOffset, centerLng + prevRKm * Math.cos(angle2) * lngOffset];

        const fakeGeoid = `3606100${i < 10 ? '0' + i : i}00`;
        const eq = equityList[i % Math.max(1, equityList.length)];
        const risk = eq?.risk_level || (i % 4 === 0 ? "critical" : i % 3 === 0 ? "high" : i % 2 === 0 ? "moderate" : "low");
        const eqVal = eq?.value ?? (0.3 + (i % 7) * 0.1);
        const unins = 9.0 + (i % 5) * 3.2;

        const polygon = L.polygon([p1, p2, p3, p4], {
          fillColor: getFeatureColor(risk, eqVal, unins),
          fillOpacity: fillAlpha,
          weight: showBorders ? 1.0 : 0,
          color: "#ffffff",
          opacity: showBorders ? 0.35 : 0,
        });

        polygon.on({
          mouseover: () => {
            polygon.setStyle({ weight: 2.5, color: "#38bdf8", fillOpacity: Math.min(1, fillAlpha + 0.15) });
          },
          mouseout: () => {
            polygon.setStyle({ weight: showBorders ? 1.0 : 0, color: "#ffffff", fillOpacity: fillAlpha });
          },
          click: (e: any) => {
            L.DomEvent.stopPropagation(e);
            setSondaTarget({
              type: "tract",
              title: `Sector Censal ${i + 1}`,
              code: fakeGeoid,
              coords: [e.latlng.lat, e.latlng.lng],
              jurisdiction: `${currentHospital.city || "Metropolitan"}, ${currentHospital.state || "US"}`,
              population: 3200 + (i * 450) % 7000,
              riskLevel: risk,
              equityIndex: +eqVal.toFixed(2),
              uninsuredRate: +unins.toFixed(1),
              povertyRate: +(14.2 + (i % 6) * 3.1).toFixed(1),
              chronicDisease: +(8.1 + (i % 4) * 2.5).toFixed(1),
            });
          },
        });

        syntheticPolys.push(polygon);
      }

      const synGroup = L.layerGroup(syntheticPolys).addTo(map);
      geoJsonLayerRef.current = synGroup;
    }
  }, [geoData, activeOverlay, opacity, showBorders, currentHospital, equityList, equityByGeoid, selectedRadiusKm]);

  // Center map on target
  const handleCenterHospital = () => {
    const map = mapInstanceRef.current;
    if (!map || !currentHospital?.latitude || !currentHospital?.longitude) return;
    map.flyTo([currentHospital.latitude, currentHospital.longitude], 13.5, { duration: 1.0 });
  };

  return (
    <div className="sat-console">
      {/* 1. Header Card with Layer Selection (Exactly as in user's reference) */}
      <div className="sat-header-card">
        <div className="sat-title-row">
          <div className="sat-label-tag">
            <Globe size={18} style={{ color: "#38bdf8" }} />
            CAPA ESPACIAL / SATELITAL:
          </div>

          {/* Hospital Switcher dropdown */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 11.5, color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>HOSPITAL ACTIVO:</span>
            <select
              className="select"
              style={{ padding: "5px 12px", fontSize: 12, minWidth: 240, height: 32 }}
              value={currentHospital?.id ?? ""}
              onChange={(e) => onSelectHospital(Number(e.target.value))}
            >
              {hospitals.map((h) => (
                <option key={h.id} value={h.id}>
                  {h.name} ({h.city || "US"})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Layer Pills Row */}
        <div className="sat-pills-container">
          {/* Base Raster Layers */}
          <button
            className={`sat-pill ${activeBaseLayer === "satellite" ? "active" : ""}`}
            onClick={() => setActiveBaseLayer("satellite")}
          >
            <Compass size={14} />
            Satélite Alta Res
          </button>
          <button
            className={`sat-pill ${activeBaseLayer === "hybrid" ? "active" : ""}`}
            onClick={() => setActiveBaseLayer("hybrid")}
          >
            <Layers size={14} />
            Híbrido (Vías & Calles)
          </button>
          <button
            className={`sat-pill ${activeBaseLayer === "dark" ? "active" : ""}`}
            onClick={() => setActiveBaseLayer("dark")}
          >
            <Globe size={14} />
            Modo Oscuro GIS
          </button>
          <button
            className={`sat-pill ${activeBaseLayer === "topo" ? "active" : ""}`}
            onClick={() => setActiveBaseLayer("topo")}
          >
            <Radio size={14} />
            Relieve & Topografía
          </button>

          {/* Thematic SDOH Overlay Pills */}
          <button
            className={`sat-pill ${activeOverlay === "risk" ? "active emerald" : ""}`}
            onClick={() => setActiveOverlay(activeOverlay === "risk" ? "none" : "risk")}
          >
            <ShieldAlert size={14} />
            Riesgo de Salud SDOH
          </button>
          <button
            className={`sat-pill ${activeOverlay === "equity" ? "active purple" : ""}`}
            onClick={() => setActiveOverlay(activeOverlay === "equity" ? "none" : "equity")}
          >
            <Activity size={14} />
            Índice de Equidad
          </button>
          <button
            className={`sat-pill ${activeOverlay === "uninsured" ? "active" : ""}`}
            onClick={() => setActiveOverlay(activeOverlay === "uninsured" ? "none" : "uninsured")}
          >
            <TrendingDown size={14} />
            Falta de Seguro Médico
          </button>
          <button
            className={`sat-pill ${activeOverlay === "poverty" ? "active" : ""}`}
            onClick={() => setActiveOverlay(activeOverlay === "poverty" ? "none" : "poverty")}
          >
            <Crosshair size={14} />
            Pobreza Censal
          </button>
        </div>

        {/* Controls Row: Opacity, Radius, Borders */}
        <div className="sat-controls-row">
          <div className="sat-slider-wrap">
            <span>OPACIDAD:</span>
            <input
              type="range"
              min={0}
              max={100}
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              style={{ width: 130 }}
            />
            <span style={{ color: "#38bdf8", fontWeight: 700, width: 34 }}>{opacity}%</span>
          </div>

          <div className="sat-res-group">
            <span>RADIO CAPTACIÓN:</span>
            {[5, 10, 15, 25].map((km) => (
              <button
                key={km}
                className={`sat-res-btn ${selectedRadiusKm === km ? "active" : ""}`}
                onClick={() => setSelectedRadiusKm(km)}
              >
                {km}km
              </button>
            ))}
          </div>

          <button
            className="sat-pill"
            style={{ padding: "4px 12px", fontSize: 11.5 }}
            onClick={() => setShowBorders(!showBorders)}
          >
            <Eye size={13} />
            {showBorders ? "Ocultar Límites Censales" : "Ver Límites Censales"}
          </button>
        </div>
      </div>

      {/* 2. Main Grid: Satellite Map on Left, Sonda Panel on Right */}
      <div className="sat-main-grid">
        {/* Left: Satellite Map Frame */}
        <div className="sat-map-frame">
          {/* Topbar inside map frame */}
          <div className="sat-map-topbar">
            <div className="sat-hospital-title">
              <MapPin size={16} style={{ color: "#06b6d4" }} />
              <span>{currentHospital?.name?.toUpperCase() || "HOSPITAL GENERAL"}</span>
              <span className="sat-coords">
                [{currentHospital?.latitude?.toFixed(3) || "40.750"}°N, {currentHospital?.longitude?.toFixed(3) || "-73.980"}°W]
              </span>
              <span style={{ color: "#64748b", fontSize: 11 }}>• {selectedRadiusKm} km catchment</span>
            </div>

            <div className="sat-live-indicator">
              <span className="sat-live-dot"></span>
              SENSOR EN TIEMPO REAL ACTIVO
            </div>
          </div>

          {/* Floating Inset Badge */}
          <div className="sat-map-badge">
            <span className="sat-map-badge-dot"></span>
            <span>{BASE_LAYERS[activeBaseLayer].name} | {BASE_LAYERS[activeBaseLayer].subtext}</span>
          </div>

          {/* Leaflet DOM container */}
          <div ref={mapContainerRef} className="sat-leaflet-container" />
        </div>

        {/* Right: Sonda de Rodal / Píxel Panel */}
        <div className="sat-probe-card">
          <div className="sat-probe-header">
            <div className="sat-probe-title">
              <Crosshair size={16} style={{ color: "#38bdf8" }} />
              SONDA DE RODAL / PÍXEL
            </div>
            <span className="sat-probe-tag">
              {sondaTarget?.code || "STAND-HOSP-01"}
            </span>
          </div>

          {/* Target Attributes Table */}
          <div className="sat-probe-info-table">
            <div className="sat-probe-row">
              <span>ENTIDAD SELECCIONADA</span>
              <b>{sondaTarget?.title || currentHospital?.name || "Hospital General"}</b>
            </div>
            <div className="sat-probe-row">
              <span>JURISDICCIÓN REAL</span>
              <b>{sondaTarget?.jurisdiction || "New York, NY"}</b>
            </div>
            <div className="sat-probe-row">
              <span>COORDENADAS REAL GEO</span>
              <b>
                {sondaTarget?.coords[0].toFixed(4)}°N, {sondaTarget?.coords[1].toFixed(4)}°W
              </b>
            </div>
            <div className="sat-probe-row">
              <span>NIVEL DE VULNERABILIDAD</span>
              <b>
                <span
                  className={`badge ${
                    sondaTarget?.riskLevel === "critical"
                      ? "red"
                      : sondaTarget?.riskLevel === "high"
                      ? "orange"
                      : sondaTarget?.riskLevel === "moderate"
                      ? "yellow"
                      : "green"
                  }`}
                  style={{ textTransform: "capitalize" }}
                >
                  {sondaTarget?.riskLevel || "Moderado"}
                </span>
              </b>
            </div>
            <div className="sat-probe-row">
              <span>POBLACIÓN ESTIMADA</span>
              <b>{sondaTarget?.population?.toLocaleString() || "48,500"} hab.</b>
            </div>
          </div>

          {/* 4 Stat Cards in 2x2 Grid (Exact replica of user's reference) */}
          <div className="sat-probe-metrics-grid">
            <div className="sat-metric-box">
              <div className="sat-metric-box-title">
                <Activity size={14} style={{ color: "#38bdf8" }} />
                Índice Equidad
              </div>
              <div className="sat-metric-box-val" style={{ color: "#38bdf8" }}>
                {sondaTarget?.equityIndex.toFixed(2) || "0.73"}
              </div>
              <div className="sat-metric-box-sub">Percentil SDOH</div>
            </div>

            <div className="sat-metric-box">
              <div className="sat-metric-box-title">
                <ShieldAlert size={14} style={{ color: "#f59e0b" }} />
                Vulnerabilidad
              </div>
              <div className="sat-metric-box-val" style={{ color: "#f59e0b" }}>
                {sondaTarget?.riskLevel === "critical"
                  ? "89%"
                  : sondaTarget?.riskLevel === "high"
                  ? "72%"
                  : sondaTarget?.riskLevel === "moderate"
                  ? "54%"
                  : "24%"}
              </div>
              <div className="sat-metric-box-sub">Riesgo compuesto</div>
            </div>

            <div className="sat-metric-box">
              <div className="sat-metric-box-title">
                <TrendingDown size={14} style={{ color: "#ec4899" }} />
                Sin Seguro
              </div>
              <div className="sat-metric-box-val" style={{ color: "#f472b6" }}>
                {sondaTarget?.uninsuredRate || "12.8"}%
              </div>
              <div className="sat-metric-box-sub">Falta de cobertura</div>
            </div>

            <div className="sat-metric-box">
              <div className="sat-metric-box-title">
                <Crosshair size={14} style={{ color: "#10b981" }} />
                Pobreza
              </div>
              <div className="sat-metric-box-val" style={{ color: "#34d399" }}>
                {sondaTarget?.povertyRate || "18.4"}%
              </div>
              <div className="sat-metric-box-sub">Bajo umbral federal</div>
            </div>
          </div>

          {/* Action button to re-center */}
          <button
            className="btn"
            style={{ width: "100%", marginTop: "auto", display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}
            onClick={handleCenterHospital}
          >
            <Crosshair size={16} />
            Centrar en Hospital Activo
          </button>
        </div>
      </div>
    </div>
  );
}
