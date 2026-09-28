import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import ProtectedRoute from "./components/ProtectedRoute";
import ErrorBoundary from "./components/ErrorBoundary";
import { useAuth } from "./store/auth";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import SdohMap from "./pages/SdohMap";
import Equity from "./pages/Equity";
import Hospitals from "./pages/Hospitals";
import Alerts from "./pages/Alerts";
import Reports from "./pages/Reports";
import Users from "./pages/Users";
import Profile from "./pages/Profile";
import CrispDm from "./pages/CrispDm";
import MlModels from "./pages/MlModels";

export default function App() {
  const { loadUser, loading, token } = useAuth();
  const { t } = useTranslation();

  useEffect(() => {
    if (token) loadUser();
    else useAuth.setState({ loading: false });
  }, [token]);

  if (loading) {
    return <div className="auth-wrap">{t("loading.app")}</div>;
  }

  return (
    <ErrorBoundary>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route element={<ProtectedRoute />}>
          <Route element={<Layout />}>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/mapa" element={<SdohMap />} />
            <Route path="/equidad" element={<Equity />} />
            <Route path="/hospitales" element={<Hospitals />} />
            <Route path="/alertas" element={<Alerts />} />
            <Route path="/reportes" element={<Reports />} />
            <Route path="/usuarios" element={<Users />} />
            <Route path="/perfil" element={<Profile />} />
            <Route path="/crisp-dm" element={<CrispDm />} />
            <Route path="/crisp-dm/:phase" element={<CrispDm />} />
            <Route path="/modelos-ml" element={<MlModels />} />
          </Route>
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </ErrorBoundary>
  );
}
