import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import {
  Building2,
  Cross,
  Layers,
  Maximize2,
  Radio,
  RotateCw,
  Tag,
  Users,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { useTheme } from "../store/theme";

export interface TwinCell {
  id: string;
  x: number;
  z: number;
  label: string;
  value: number; // 0-1
  risk: string;
  percentile?: number;
  population?: number;
  name?: string;
}

export interface TwinHospital {
  id: number;
  name: string;
  city?: string;
  state?: string;
}

const RISK_COLORS: Record<string, string> = {
  low: "#22c55e",
  moderate: "#f59e0b",
  high: "#f97316",
  critical: "#ef4444",
};

const VALUE_STOPS = ["#06b6d4", "#3b82f6", "#8b5cf6", "#e879f9"];

const TRACT_SPACING = 1.7;
const CORE_CLEARANCE = 3.6;
const MAX_RINGS = 48;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
const TOP_LABEL_COUNT = 36;

function clamp01(v: number) {
  return Math.min(1, Math.max(0, v));
}

export function colorForRisk(level: string): THREE.Color {
  return new THREE.Color(RISK_COLORS[level] || RISK_COLORS.low);
}

export function colorForValue(value: number): THREE.Color {
  const t = clamp01(value) * (VALUE_STOPS.length - 1);
  const i = Math.min(Math.floor(t), VALUE_STOPS.length - 2);
  return new THREE.Color(VALUE_STOPS[i]).lerp(
    new THREE.Color(VALUE_STOPS[i + 1]),
    t - i
  );
}

function buildRadialLayout(count: number): { x: number; z: number }[] {
  const positions: { x: number; z: number }[] = [];
  if (count <= 0) return positions;

  const radii: number[] = [];
  let capacity = 0;
  let r = CORE_CLEARANCE;
  while (capacity < count && radii.length < MAX_RINGS) {
    radii.push(r);
    capacity += Math.max(1, Math.floor((2 * Math.PI * r) / TRACT_SPACING));
    r += TRACT_SPACING;
  }

  let index = 0;
  radii.forEach((radius, ring) => {
    const cap = Math.max(1, Math.floor((2 * Math.PI * radius) / TRACT_SPACING));
    const placed = Math.min(cap, count - index);
    if (placed <= 0) return;
    const offset = ring * GOLDEN_ANGLE;
    for (let i = 0; i < placed; i++) {
      const theta = offset + (i / placed) * Math.PI * 2;
      positions.push({ x: Math.cos(theta) * radius, z: Math.sin(theta) * radius });
      index++;
    }
  });

  return positions;
}

/** Progresivo por altura: 0 → structured, 1 → DATA, 2 → RADAR. */
function floorTexture(isLight: boolean): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d")!;

  ctx.clearRect(0, 0, size, size);
  ctx.strokeStyle = isLight ? "rgba(37,99,235,0.10)" : "rgba(96,165,250,0.16)";
  ctx.lineWidth = 1.5;
  for (let i = 0; i < 6; i++) {
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, (size / 2) * (0.16 + i * 0.16), 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.strokeStyle = isLight ? "rgba(37,99,235,0.07)" : "rgba(148,163,184,0.10)";
  for (let i = 0; i < 24; i++) {
    const theta = (i / 24) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(size / 2, size / 2);
    ctx.lineTo(
      size / 2 + Math.cos(theta) * size * 0.5,
      size / 2 + Math.sin(theta) * size * 0.5
    );
    ctx.stroke();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function groundTexture(isLight: boolean): THREE.CanvasTexture {
  const size = 512;
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  if (isLight) {
    gradient.addColorStop(0, "#e6edf7");
    gradient.addColorStop(0.45, "#c6d2e4");
    gradient.addColorStop(1, "#8fa1bd");
  } else {
    gradient.addColorStop(0, "#13294a");
    gradient.addColorStop(0.45, "#0a1526");
    gradient.addColorStop(1, "#03060d");
  }
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/* ------------------------------------------------------------------ */
/* Tract bar                                                           */
/* ------------------------------------------------------------------ */

function TractCell({
  cell,
  color,
  accent,
  dimmed,
  onHover,
  onSelect,
  hovered,
  selected,
  showLabel,
  isWireframe,
  isLight,
  growthKey,
  shadows,
}: {
  cell: TwinCell;
  color: THREE.Color;
  accent: THREE.Color;
  dimmed: boolean;
  onHover: (cell: TwinCell | null) => void;
  onSelect: (cell: TwinCell) => void;
  hovered: boolean;
  selected: boolean;
  showLabel: boolean;
  isWireframe: boolean;
  isLight: boolean;
  growthKey: number;
  shadows: boolean;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const glowRef = useRef<THREE.Mesh>(null);
  const pulseRef = useRef<THREE.Mesh>(null);
  const growth = useRef(0);
  const prevKey = useRef(growthKey);

  const height = 0.45 + clamp01(cell.value) * 8.4;
  const isActive = hovered || selected;
  const opacity = dimmed ? 0.24 : 1;
  const bodyColor = useMemo(
    () => (dimmed ? color.clone().multiplyScalar(0.45) : color),
    [color, dimmed]
  );

  if (prevKey.current !== growthKey) {
    prevKey.current = growthKey;
    growth.current = 0;
  }

  useFrame((state, delta) => {
    const g = groupRef.current;
    if (!g) return;

    if (growth.current < 1) {
      growth.current = Math.min(1, growth.current + delta * 1.5);
    }
    const eased = 1 - Math.pow(1 - growth.current, 3);
    const wave = Math.sin(state.clock.elapsedTime * 1.1 + cell.x * 0.35 + cell.z * 0.45);

    g.scale.y = Math.max(0.001, eased);
    g.position.y = isActive ? wave * 0.09 : wave * 0.03;

    if (glowRef.current) {
      const mat = glowRef.current.material as THREE.MeshBasicMaterial;
      mat.opacity = (isActive ? 0.55 : hovered || selected ? 0.3 : 0.14) * opacity;
      const target = isActive ? 1 : hovered || selected ? 1.18 : 1;
      glowRef.current.scale.x += (target - glowRef.current.scale.x) * 0.15;
      glowRef.current.scale.y += (target - glowRef.current.scale.y) * 0.15;
    }

    if (pulseRef.current) {
      const t = (state.clock.elapsedTime * 0.55) % 1;
      pulseRef.current.scale.setScalar(0.9 + t * 1.5);
      (pulseRef.current.material as THREE.MeshBasicMaterial).opacity =
        (selected ? 0.5 : 0.25) * (1 - t);
    }
  });

  const emissive = isActive ? accent : new THREE.Color(color).multiplyScalar(0.18);

  return (
    <group position={[cell.x, 0, cell.z]}>
      {/* Suelo del tract */}
      <mesh position={[0, 0.012, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow={shadows}>
        <planeGeometry args={[1.42, 1.42]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={isActive ? 0.34 : dimmed ? 0.04 : 0.09}
          depthWrite={false}
        />
      </mesh>

      <group
        ref={groupRef}
        onPointerOver={(e) => {
          e.stopPropagation();
          onHover(cell);
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={(e) => {
          e.stopPropagation();
          onHover(null);
          document.body.style.cursor = "auto";
        }}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(cell);
        }}
      >
        <mesh position={[0, height / 2, 0]} castShadow={shadows} receiveShadow={shadows}>
          <boxGeometry args={[1.02, height, 1.02]} />
          <meshStandardMaterial
            color={bodyColor}
            roughness={isLight ? 0.28 : 0.2}
            metalness={isLight ? 0.25 : 0.5}
            emissive={emissive}
            emissiveIntensity={dimmed ? 0.05 : isActive ? 0.9 : 0.35}
            wireframe={isWireframe}
          />
        </mesh>

        {/* Corona superior: indica el valor del índice */}
        <mesh position={[0, height + 0.05, 0]}>
          <boxGeometry args={[1.12, 0.1, 1.12]} />
          <meshStandardMaterial
            color={bodyColor}
            emissive={isActive ? accent : color}
            emissiveIntensity={dimmed ? 0.08 : isActive ? 1.8 : 0.9}
            roughness={0.25}
            metalness={0.4}
          />
        </mesh>
      </group>

      {/* Halo en el suelo */}
      <mesh ref={glowRef} position={[0, 0.02, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.62, 0.86, 40]} />
        <meshBasicMaterial
          color={isActive ? accent : color}
          transparent
          opacity={0.14 * opacity}
          depthWrite={false}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Pulso expansivo para la selección */}
      {isActive && (
        <mesh ref={pulseRef} position={[0, 0.03, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.7, 0.78, 40]} />
          <meshBasicMaterial
            color={accent}
            transparent
            depthWrite={false}
            side={THREE.DoubleSide}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      )}

      {/* Baliza vertical del tract seleccionado */}
      {selected && (
        <mesh position={[0, height / 2 + 5, 0]}>
          <cylinderGeometry args={[0.09, 0.16, 10, 12, 1, true]} />
          <meshBasicMaterial
            color={accent}
            transparent
            opacity={0.35}
            depthWrite={false}
            side={THREE.DoubleSide}
            blending={THREE.AdditiveBlending}
          />
        </mesh>
      )}

      {showLabel && (
        <Text
          position={[0, height + 0.72, 0]}
          fontSize={0.34}
          color={isActive ? (isLight ? "#1d4ed8" : "#bfdbfe") : isLight ? "#0f172a" : "#e2e8f0"}
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.035}
          outlineColor={isLight ? "#f8fafc" : "#030712"}
          maxWidth={1.6}
        >
          {cell.label}
        </Text>
      )}
    </group>
  );
}

/* ------------------------------------------------------------------ */
/* Núcleo hospitalario                                                 */
/* ------------------------------------------------------------------ */

function HospitalCore({
  isLight,
  accent,
  onSelect,
  selected,
}: {
  isLight: boolean;
  accent: THREE.Color;
  onSelect: () => void;
  selected: boolean;
}) {
  const towerRef = useRef<THREE.Group>(null);
  const ringA = useRef<THREE.Mesh>(null);
  const ringB = useRef<THREE.Mesh>(null);
  const beamRef = useRef<THREE.Mesh>(null);
  const haloRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    if (towerRef.current) towerRef.current.rotation.y = t * 0.18;
    if (ringA.current) ringA.current.rotation.z = t * 0.35;
    if (ringB.current) ringB.current.rotation.z = -t * 0.22;
    if (beamRef.current) {
      (beamRef.current.material as THREE.MeshBasicMaterial).opacity =
        (selected ? 0.2 : 0.11) + Math.sin(t * 1.6) * 0.03;
    }
    if (haloRef.current) {
      const k = (t * 0.35) % 1;
      haloRef.current.scale.setScalar(1 + k * 2.6);
      (haloRef.current.material as THREE.MeshBasicMaterial).opacity =
        (selected ? 0.35 : 0.18) * (1 - k);
    }
  });

  const shell = isLight ? "#cbd5e1" : "#1e293b";
  const shellLight = isLight ? "#e2e8f0" : "#334155";

  return (
    <group>
      {/* Baliza de luz */}
      <mesh ref={beamRef} position={[0, 15, 0]}>
        <cylinderGeometry args={[0.7, 2.6, 30, 24, 1, true]} />
        <meshBasicMaterial
          color={accent}
          transparent
          opacity={0.11}
          depthWrite={false}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      {/* Onda expansiva en el suelo */}
      <mesh ref={haloRef} position={[0, 0.03, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1, 1.08, 64]} />
        <meshBasicMaterial
          color={accent}
          transparent
          depthWrite={false}
          side={THREE.DoubleSide}
          blending={THREE.AdditiveBlending}
        />
      </mesh>

      <group
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          document.body.style.cursor = "auto";
        }}
      >
        <group ref={towerRef}>
          {/* Zócalo */}
          <mesh position={[0, 0.18, 0]} castShadow receiveShadow>
            <cylinderGeometry args={[2.5, 2.9, 0.36, 8]} />
            <meshStandardMaterial
              color={shell}
              roughness={0.55}
              metalness={0.35}
              emissive={accent}
              emissiveIntensity={0.15}
            />
          </mesh>

          {/* Cuerpo principal */}
          <mesh position={[0, 1.95, 0]} castShadow receiveShadow>
            <boxGeometry args={[2.9, 3.2, 2.9]} />
            <meshStandardMaterial
              color={shell}
              roughness={0.4}
              metalness={0.5}
              emissive={accent}
              emissiveIntensity={0.12}
            />
          </mesh>

          {/* Cintura de ventanas */}
          {[1.35, 2.05, 2.75].map((y) => (
            <mesh key={y} position={[0, y, 0]}>
              <boxGeometry args={[2.98, 0.13, 2.98]} />
              <meshStandardMaterial
                color={accent}
                emissive={accent}
                emissiveIntensity={1.1}
                roughness={0.3}
                metalness={0.2}
              />
            </mesh>
          ))}

          {/* Torre superior */}
          <mesh position={[0, 4.3, 0]} castShadow>
            <boxGeometry args={[2, 1.5, 2]} />
            <meshStandardMaterial
              color={shellLight}
              roughness={0.35}
              metalness={0.55}
              emissive={accent}
              emissiveIntensity={0.18}
            />
          </mesh>

          <mesh position={[0, 5.35, 0]}>
            <boxGeometry args={[2.06, 0.12, 2.06]} />
            <meshStandardMaterial
              color={accent}
              emissive={accent}
              emissiveIntensity={1.4}
              roughness={0.25}
            />
          </mesh>

          {/* Cruz healthcare */}
          <mesh position={[0, 6.1, 0]}>
            <boxGeometry args={[1.5, 0.42, 0.2]} />
            <meshStandardMaterial color="#f8fafc" emissive="#ffffff" emissiveIntensity={0.7} roughness={0.4} />
          </mesh>
          <mesh position={[0, 6.1, 0]}>
            <boxGeometry args={[0.42, 1.4, 0.2]} />
            <meshStandardMaterial color="#f8fafc" emissive="#ffffff" emissiveIntensity={0.7} roughness={0.4} />
          </mesh>

          {/* Antena */}
          <mesh position={[0, 7.4, 0]}>
            <cylinderGeometry args={[0.05, 0.05, 1.5, 8]} />
            <meshStandardMaterial color={accent} emissive={accent} emissiveIntensity={1.2} metalness={0.8} roughness={0.2} />
          </mesh>
          <mesh position={[0, 8.25, 0]}>
            <sphereGeometry args={[0.16, 16, 16]} />
            <meshBasicMaterial color={selected ? "#ffffff" : accent} />
          </mesh>
        </group>
      </group>

      {/* Anillos orbitales */}
      <mesh ref={ringA} position={[0, 0.05, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[3.1, 3.28, 72, 1, 0, Math.PI * 1.35]} />
        <meshBasicMaterial
          color={accent}
          transparent
          opacity={isLight ? 0.65 : 0.5}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <mesh ref={ringB} position={[0, 0.06, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[3.55, 3.66, 72, 1, Math.PI, Math.PI * 1.6]} />
        <meshBasicMaterial
          color="#818cf8"
          transparent
          opacity={isLight ? 0.55 : 0.4}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>

      <Text
        position={[0, 9.3, 0]}
        fontSize={0.46}
        color={isLight ? "#0f172a" : "#f1f5f9"}
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.045}
        outlineColor={isLight ? "#f8fafc" : "#030712"}
        maxWidth={6}
      >
        HOSPITAL
      </Text>
    </group>
  );
}

/* ------------------------------------------------------------------ */
/* Escenario                                                            */
/* ------------------------------------------------------------------ */

function CityFloor({ radius, isLight }: { radius: number; isLight: boolean }) {
  const gridRef = useRef<THREE.GridHelper>(null);
  const floorTex = useMemo(() => floorTexture(isLight), [isLight]);
  const groundTex = useMemo(() => groundTexture(isLight), [isLight]);
  const size = Math.max(radius * 2.5, 40);

  useEffect(() => {
    const grid = gridRef.current;
    if (!grid) return;
    const mat = grid.material as THREE.Material;
    mat.transparent = true;
    mat.opacity = isLight ? 0.35 : 0.22;
  }, [isLight, radius]);

  useEffect(() => () => floorTex.dispose(), [floorTex]);
  useEffect(() => () => groundTex.dispose(), [groundTex]);

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.06, 0]} receiveShadow>
        <circleGeometry args={[radius * 1.35 + 9, 96]} />
        <meshStandardMaterial
          map={groundTex}
          roughness={isLight ? 0.85 : 0.42}
          metalness={isLight ? 0.05 : 0.65}
        />
      </mesh>

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]}>
        <circleGeometry args={[radius * 1.3 + 6, 96]} />
        <meshBasicMaterial map={floorTex} transparent opacity={isLight ? 0.9 : 0.75} depthWrite={false} />
      </mesh>

      <gridHelper
        ref={gridRef}
        args={[size, Math.round(size / TRACT_SPACING), isLight ? "#2563eb" : "#3b82f6", isLight ? "#94a3b8" : "#1e3a5f"]}
        position={[0, 0, 0]}
      />

      {/* Borde luminoso del perímetro */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.008, 0]}>
        <ringGeometry args={[radius * 1.3 + 5.4, radius * 1.3 + 5.7, 128]} />
        <meshBasicMaterial
          color="#3b82f6"
          transparent
          opacity={isLight ? 0.45 : 0.35}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

function DataMotes({ radius, isLight }: { radius: number; isLight: boolean }) {
  const ref = useRef<THREE.Points>(null);
  const count = 420;

  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = CORE_CLEARANCE + Math.random() * (radius * 1.7 + 8);
      arr[i * 3] = Math.cos(angle) * r;
      arr[i * 3 + 1] = Math.random() * 26 + 0.5;
      arr[i * 3 + 2] = Math.sin(angle) * r;
    }
    return arr;
  }, [radius]);

  useFrame((state) => {
    if (ref.current) ref.current.rotation.y = state.clock.elapsedTime * 0.012;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.16}
        color={isLight ? "#3b82f6" : "#7dd3fc"}
        transparent
        opacity={isLight ? 0.35 : 0.5}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

function CameraRig({ radius, resetKey }: { radius: number; resetKey: number }) {
  const { camera } = useThree();
  const goal = useRef(new THREE.Vector3(18, 16, 18));
  const frames = useRef(0);
  const targetY = Math.min(radius * 0.16, 4);

  useEffect(() => {
    const dist = Math.min(radius * 1.15 + 16, 52);
    goal.current.set(dist * 0.66, dist * 0.62, dist * 0.66);
    frames.current = 90;
  }, [radius, resetKey]);

  useFrame(() => {
    if (frames.current <= 0) return;
    frames.current -= 1;
    camera.position.lerp(goal.current, 0.09);
    camera.lookAt(0, targetY, 0);
  });

  return null;
}

/* ------------------------------------------------------------------ */
/* Componente principal                                                 */
/* ------------------------------------------------------------------ */

interface Twin3DProps {
  cells: TwinCell[];
  colorBy: "value" | "risk";
  /** Texto del HUD cuando el color no es el nivel de riesgo del índice (p. ej. predicción ML). */
  hudLabel?: string;
  onSelect: (cell: TwinCell) => void;
  hospital?: TwinHospital | null;
  loading?: boolean;
}

export default function Twin3D({ cells, colorBy, onSelect, hospital, loading, hudLabel }: Twin3DProps) {
  const { theme } = useTheme();
  const isLight = theme === "light";
  const { t } = useTranslation();

  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [autoRotate, setAutoRotate] = useState(false);
  const [showLabels, setShowLabels] = useState(false);
  const [isWireframe, setIsWireframe] = useState(false);
  const [resetKey, setResetKey] = useState(0);

  const accent = useMemo(
    () => new THREE.Color(isLight ? "#2563eb" : "#60a5fa"),
    [isLight]
  );

  const positioned = useMemo(() => {
    const sorted = [...cells].sort((a, b) => b.value - a.value);
    const layout = buildRadialLayout(sorted.length);
    return sorted.map((cell, i) => ({ ...cell, ...layout[i] }));
  }, [cells]);

  const radius = useMemo(
    () => (positioned.length ? Math.max(...positioned.map((p) => Math.hypot(p.x, p.z))) : 18),
    [positioned]
  );

  const labelIds = useMemo(() => {
    const ids = new Set<string>();
    if (!showLabels) return ids;
    [...cells]
      .sort((a, b) => b.value - a.value)
      .slice(0, TOP_LABEL_COUNT)
      .forEach((c) => ids.add(c.id));
    return ids;
  }, [cells, showLabels]);

  const summary = useMemo(() => {
    const counts: Record<string, number> = { low: 0, moderate: 0, high: 0, critical: 0 };
    let sum = 0;
    let population = 0;
    cells.forEach((c) => {
      counts[c.risk] = (counts[c.risk] || 0) + 1;
      sum += c.value;
      population += c.population ?? 0;
    });
    return {
      counts,
      avg: cells.length ? sum / cells.length : 0,
      population,
      criticalPct: cells.length ? (counts.critical / cells.length) * 100 : 0,
    };
  }, [cells]);

  const hoveredCell = useMemo(
    () => cells.find((c) => c.id === hoveredId) ?? null,
    [cells, hoveredId]
  );

  const shadows = positioned.length <= 420;
  const growthKey = resetKey + (hospital?.id ?? 0);

  const handleSelect = (cell: TwinCell) => {
    setSelectedId(cell.id);
    onSelect(cell);
  };

  const handleCoreSelect = () => {
    setSelectedId(null);
  };

  return (
    <div className="twin-canvas-wrapper">
      <Canvas
        className="twin-canvas"
        shadows={shadows}
        dpr={[1, 1.75]}
        camera={{ position: [20, 18, 20], fov: 42 }}
        gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
        onPointerMissed={() => setSelectedId(null)}
      >
        <color attach="background" args={[isLight ? "#cfdaea" : "#04070f"]} />
        <fog
          attach="fog"
          args={[isLight ? "#cfdaea" : "#04070f", radius * 0.85 + 12, radius * 2.6 + 46]}
        />

        <hemisphereLight
          args={[isLight ? "#ffffff" : "#93c5fd", isLight ? "#8ba0bd" : "#0b1220", isLight ? 0.8 : 0.7]}
        />
        <ambientLight intensity={isLight ? 0.32 : 0.28} />
        <directionalLight
          position={[radius, radius * 1.7, radius * 0.7]}
          intensity={isLight ? 1.5 : 1.15}
          color={isLight ? "#ffffff" : "#dbeafe"}
          castShadow={shadows}
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
          shadow-camera-near={1}
          shadow-camera-far={radius * 4 + 40}
          shadow-camera-left={-radius - 10}
          shadow-camera-right={radius + 10}
          shadow-camera-top={radius + 10}
          shadow-camera-bottom={-radius - 10}
          shadow-bias={-0.0008}
        />
        <pointLight
          position={[-radius * 0.7, 12, -radius * 0.7]}
          intensity={Math.pow(radius * 1.5, 2) * 0.5}
          distance={radius * 3 + 30}
          decay={2}
          color="#6366f1"
        />
        <pointLight
          position={[radius * 0.6, 9, -radius * 0.5]}
          intensity={Math.pow(radius * 1.2, 2) * 0.35}
          distance={radius * 2.6 + 24}
          decay={2}
          color={isLight ? "#a5b4fc" : "#38bdf8"}
        />
        <pointLight position={[0, 7, 0]} intensity={90} distance={30} decay={2} color={isLight ? "#93c5fd" : "#60a5fa"} />

        <OrbitControls
          enablePan
          enableZoom
          enableDamping
          dampingFactor={0.07}
          minDistance={8}
          maxDistance={radius * 3 + 40}
          target={[0, Math.min(radius * 0.16, 4), 0]}
          autoRotate={autoRotate}
          autoRotateSpeed={0.9}
          maxPolarAngle={Math.PI / 2 - 0.04}
        />

        <CameraRig radius={radius} resetKey={resetKey} />
        <CityFloor radius={radius} isLight={isLight} />
        <DataMotes radius={radius} isLight={isLight} />
        <HospitalCore
          isLight={isLight}
          accent={accent}
          selected={!selectedId}
          onSelect={handleCoreSelect}
        />

        {positioned.map((cell) => {
          const color = colorBy === "risk" ? colorForRisk(cell.risk) : colorForValue(cell.value);
          const isSel = selectedId === cell.id;
          const isHov = hoveredId === cell.id;
          return (
            <TractCell
              key={cell.id}
              cell={cell}
              color={color}
              accent={accent}
              dimmed={!!selectedId && !isSel && !isHov}
              onHover={(c) => setHoveredId(c?.id ?? null)}
              onSelect={handleSelect}
              hovered={isHov}
              selected={isSel}
              showLabel={labelIds.has(cell.id) || isSel || isHov}
              isWireframe={isWireframe}
              isLight={isLight}
              growthKey={growthKey}
              shadows={shadows}
            />
          );
        })}

        {hoveredCell && (
          <Html
            position={[
              positioned.find((p) => p.id === hoveredCell.id)?.x ?? 0,
              0.45 + clamp01(hoveredCell.value) * 8.4 + 1.6,
              positioned.find((p) => p.id === hoveredCell.id)?.z ?? 0,
            ]}
            center
            distanceFactor={40}
            zIndexRange={[9, 1]}
            style={{ pointerEvents: "none" }}
          >
            <div className="twin-tooltip">
              <div className="twin-tooltip-head">
                <span
                  className="twin-tooltip-dot"
                  style={{ background: colorBy === "risk" ? RISK_COLORS[hoveredCell.risk] : "#8b5cf6" }}
                />
                {hoveredCell.name || hoveredCell.label}
              </div>
              <div className="twin-tooltip-grid">
                <span>{t("dashboard.equityIndex")}</span>
                <b>{hoveredCell.value.toFixed(3)}</b>
                <span>{t("dashboard.percentile")}</span>
                <b>{hoveredCell.percentile != null ? `${hoveredCell.percentile.toFixed(1)}%` : "—"}</b>
                <span>{t("dashboard.population")}</span>
                <b>
                  {hoveredCell.population != null
                    ? hoveredCell.population.toLocaleString()
                    : "—"}
                </b>
              </div>
              <div className={`twin-tooltip-risk ${hoveredCell.risk}`}>
                {t(`common.${hoveredCell.risk}`)}
              </div>
            </div>
          </Html>
        )}
      </Canvas>

      {/* ---------- HUD superior ---------- */}
      <div className="twin-hud-overlay">
        <div className="twin-hud-left">
          <div className="twin-hud-badge">
            <Cross size={13} style={{ color: isLight ? "#2563eb" : "#93c5fd" }} />
            <span>{hospital?.name ?? t("dashboard.twinFallbackName")}</span>
            {hospital?.city && (
              <span className="twin-hud-sub">
                {hospital.city}
                {hospital.state ? `, ${hospital.state}` : ""}
              </span>
            )}
          </div>
          <div className="twin-hud-badge twin-hud-badge-dim">
            <span className="status-dot" />
            <span>
              {hudLabel ?? (colorBy === "risk" ? t("dashboard.coloredByRisk") : t("dashboard.coloredByEquity"))}
            </span>
            <span className="twin-hud-sub">· {cells.length} {t("dashboard.tractsLoaded")}</span>
          </div>
        </div>

        <div className="twin-controls-toolbar">
          <button
            type="button"
            className={`btn btn-sm ${autoRotate ? "" : "secondary"}`}
            title={t("dashboard.toggleRotation")}
            onClick={() => setAutoRotate((r) => !r)}
          >
            <RotateCw size={14} className={autoRotate ? "spin-icon" : ""} />
            <span>{autoRotate ? t("dashboard.pause") : t("dashboard.rotate")}</span>
          </button>
          <button
            type="button"
            className={`btn btn-sm ${showLabels ? "" : "secondary"}`}
            title={t("dashboard.labelsHint")}
            onClick={() => setShowLabels((l) => !l)}
          >
            <Tag size={14} />
            <span>{t("dashboard.labels")}</span>
          </button>
          <button
            type="button"
            className={`btn btn-sm ${isWireframe ? "" : "secondary"}`}
            title={t("dashboard.wireframe")}
            onClick={() => setIsWireframe((w) => !w)}
          >
            <Layers size={14} />
          </button>
          <button
            type="button"
            className="btn btn-sm secondary"
            title={t("dashboard.centerCamera")}
            onClick={() => setResetKey((k) => k + 1)}
          >
            <Maximize2 size={14} />
            <span>{t("dashboard.center")}</span>
          </button>
        </div>
      </div>

      {/* ---------- Panel de métricas ---------- */}
      <div className="twin-metrics-overlay">
        <div className="twin-metrics-title">
          <Radio size={13} />
          {t("dashboard.catchmentProfile")}
        </div>
        <div className="twin-metric">
          <Building2 size={13} />
          <span>{t("dashboard.tracts")}</span>
          <b>{cells.length}</b>
        </div>
        <div className="twin-metric">
          <Users size={13} />
          <span>{t("dashboard.population")}</span>
          <b>{summary.population.toLocaleString()}</b>
        </div>
        <div className="twin-metric">
          <span className="twin-metric-bar" />
          <span>{t("dashboard.vulnerabilityIndex")}</span>
          <b>{summary.avg.toFixed(3)}</b>
        </div>
        <div className="twin-risk-bars">
          {(["critical", "high", "moderate", "low"] as const).map((key) => {
            const total = Object.values(summary.counts).reduce((a, b) => a + b, 0);
            const pct = total ? (summary.counts[key] / total) * 100 : 0;
            return (
              <div className="twin-risk-row" key={key}>
                <span className="twin-risk-name" style={{ color: RISK_COLORS[key] }}>
                  {t(`common.${key}`)}
                </span>
                <span className="twin-risk-track">
                  <span
                    className="twin-risk-fill"
                    style={{ width: `${pct}%`, background: RISK_COLORS[key] }}
                  />
                </span>
                <b>{summary.counts[key]}</b>
              </div>
            );
          })}
        </div>
      </div>

      {/* ---------- Leyenda ---------- */}
      <div className="twin-legend-overlay">
        {colorBy === "risk" ? (
          <>
            <div className="twin-legend-title">{t("dashboard.sdohRiskScale")}</div>
            <div className="twin-legend-items">
              {(["low", "moderate", "high", "critical"] as const).map((key) => (
                <span className="twin-legend-item" key={key}>
                  <span className="twin-legend-dot" style={{ background: RISK_COLORS[key] }} />
                  {t(`common.${key}`)}
                </span>
              ))}
            </div>
          </>
        ) : (
          <>
            <div className="twin-legend-title">{t("dashboard.vulnerabilityIndex")}</div>
            <div className="legend-scale-bar" style={{ background: `linear-gradient(90deg, ${VALUE_STOPS.join(", ")})` }} />
            <div className="twin-legend-scale">
              <span>0.0 · {t("dashboard.favorable")}</span>
              <span>1.0 · {t("dashboard.vulnerable")}</span>
            </div>
          </>
        )}
        <div className="twin-legend-note">{t("dashboard.radiusNote")}</div>
      </div>

      {loading && (
        <div className="twin-loading-overlay">
          <div className="twin-loading-ring" />
          <span>{t("dashboard.loadingTwin")}</span>
        </div>
      )}
    </div>
  );
}
