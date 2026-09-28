// Utility to ensure Leaflet (window.L) is ready to use in Vite / React
export function loadLeaflet(): Promise<any> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("Leaflet can only be loaded in a browser environment"));
  }

  const win = window as any;
  if (win.L && typeof win.L.map === "function") {
    return Promise.resolve(win.L);
  }

  return new Promise((resolve, reject) => {
    // Check if script is already present in DOM
    const existingScript = document.querySelector('script[src*="leaflet"]') as HTMLScriptElement | null;
    const existingLink = document.querySelector('link[href*="leaflet"]') as HTMLLinkElement | null;

    if (!existingLink) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
      link.crossOrigin = "";
      document.head.appendChild(link);
    }

    if (existingScript) {
      if (win.L) {
        resolve(win.L);
        return;
      }
      existingScript.addEventListener("load", () => resolve((window as any).L));
      existingScript.addEventListener("error", () => reject(new Error("Failed to load Leaflet script")));
      return;
    }

    const script = document.createElement("script");
    script.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
    script.crossOrigin = "";
    script.onload = () => {
      resolve((window as any).L);
    };
    script.onerror = () => {
      reject(new Error("Failed to load Leaflet from CDN"));
    };
    document.head.appendChild(script);
  });
}
