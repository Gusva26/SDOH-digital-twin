export interface Role {
  id: number;
  code: string;
  name: string;
  description?: string;
  is_system: boolean;
}

export interface Profile {
  id: number;
  user_id: number;
  full_name?: string;
  job_title?: string;
  department?: string;
  organization?: string;
  phone?: string;
}

export interface User {
  id: number;
  email: string;
  username: string;
  is_active: boolean;
  is_superuser: boolean;
  last_login?: string;
  created_at: string;
  roles: Role[];
  profile?: Profile;
}

export interface Indicator {
  id: number;
  code: string;
  name: string;
  domain?: string;
  source?: string;
  unit?: string;
  threshold_alert?: number;
}

export interface Alert {
  id: number;
  severity?: string;
  status: string;
  indicator_code?: string;
  indicator_name?: string;
  observed_value?: number;
  message?: string;
  created_at?: string;
}

export interface EquityIndex {
  id: number;
  tract_id: number;
  tract_geoid?: string;
  year: number;
  index_type: string;
  value: number;
  percentile?: number;
  risk_level?: string;
}

export interface Hospital {
  id: number;
  name: string;
  slug?: string;
  cms_id?: string;
  address?: string;
  city?: string;
  state?: string;
  zip_code?: string;
  description?: string;
  latitude?: number;
  longitude?: number;
}

export interface Catchment {
  id: number;
  hospital_id: number;
  name: string;
  catchment_type: string;
  radius_km?: number;
  tract_count?: number;
}

export interface GeoFeature {
  type: string;
  properties: {
    geoid: string;
    name?: string;
    population?: number;
    [key: string]: unknown;
  };
  geometry?: Record<string, any> | null;
}

export interface CrispPhase {
  key: string;
  order: number;
  roman: string;
  name: string;
  name_en: string;
  layer: string;
  objectives: string[];
  question: string;
  ready: boolean;
}

export interface CrispPhaseOverview {
  year: number;
  phases: CrispPhase[];
  completed: number;
  total: number;
  counts: {
    tracts: number;
    indicators: number;
    values: number;
    computed_indexes: number;
    alerts: number;
    audit_events: number;
  };
}
