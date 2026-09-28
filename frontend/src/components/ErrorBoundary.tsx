import { Component, ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }
      return (
        <div
          style={{
            padding: 32,
            margin: "40px auto",
            maxWidth: 620,
            background: "var(--bg-surface)",
            border: "1px solid var(--border-subtle)",
            borderRadius: 12,
            boxShadow: "0 10px 25px rgba(0,0,0,0.2)",
          }}
        >
          <h3 style={{ color: "#ef4444", marginTop: 0, fontSize: 18 }}>
            Error al renderizar la sección
          </h3>
          <p style={{ color: "var(--text-muted)", fontSize: 14, lineHeight: 1.6 }}>
            {this.state.error?.message || "Ocurrió un error inesperado al cargar la vista."}
          </p>
          <div style={{ display: "flex", gap: 12, marginTop: 20 }}>
            <button className="btn" type="button" onClick={this.handleReset}>
              Reintentar
            </button>
            <button
              className="btn btn-secondary"
              type="button"
              onClick={() => window.location.reload()}
            >
              Recargar página
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
