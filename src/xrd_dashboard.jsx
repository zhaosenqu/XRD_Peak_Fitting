import { useState, useEffect, useCallback, useRef } from "react";
import {
  Scatter, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceArea, ReferenceLine, ComposedChart
} from "recharts";

// ─── Palette (light academic) ────────────────────────────────────────────────
const C = {
  bg: "#f6f7f9",
  panel: "#ffffff",
  border: "#e3e6ea",
  borderStrong: "#cbd2d9",
  text: "#1f2933",
  textMuted: "#6b7280",
  textFaint: "#9aa5b1",
  accent: "#2563eb",
  accentSoft: "#eef3fe",
  marker: "#d97706",
  grid: "#edeff2",
  good: "#047857",
  warn: "#b45309",
  bad: "#b91c1c",
};
const SANS = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";

// Physical constants (lengths in Å)
const R_E = 2.8179403262e-5;   // classical electron radius, Å
const N_A = 6.02214076e23;     // Avogadro number, /mol
const DEG = Math.PI / 180;

// ─── Pseudo-Voigt Math ───────────────────────────────────────────────────────
const gaussian = (x, x0, fwhm) => Math.exp(-4 * Math.LN2 * ((x - x0) ** 2) / (fwhm ** 2));
const lorentzian = (x, x0, fwhm) => 1 / (1 + 4 * ((x - x0) ** 2) / (fwhm ** 2));
const pseudoVoigt = (x, params) => {
  const [I0, x0, fwhm, eta, a, b] = params;
  const G = gaussian(x, x0, fwhm);
  const L = lorentzian(x, x0, fwhm);
  return I0 * (eta * L + (1 - eta) * G) + a + b * x;
};

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const median = (arr) => {
  const s = [...arr].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const smooth = (arr, half) => arr.map((_, i) => {
  let s = 0, c = 0;
  for (let k = -half; k <= half; k++) { const j = i + k; if (j >= 0 && j < arr.length) { s += arr[j]; c++; } }
  return s / c;
});

// Levenberg-Marquardt fitting
function fitPseudoVoigt(xData, yData, initialParams, maxIter = 800) {
  let params = [...initialParams];
  const n = xData.length;
  const np = params.length;
  let lambda = 0.001;
  const delta = 1e-7;

  const residuals = (p) => xData.map((x, i) => yData[i] - pseudoVoigt(x, p));
  const chi2 = (p) => residuals(p).reduce((s, r) => s + r * r, 0);

  for (let iter = 0; iter < maxIter; iter++) {
    const res = residuals(params);
    const J = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < np; j++) {
        const pPlus = [...params];
        const h = Math.max(Math.abs(params[j]) * delta, delta);
        pPlus[j] += h;
        row.push((pseudoVoigt(xData[i], pPlus) - pseudoVoigt(xData[i], params)) / h);
      }
      J.push(row);
    }
    // J^T * J and J^T * r
    const JtJ = Array.from({ length: np }, () => new Float64Array(np));
    const JtR = new Float64Array(np);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < np; j++) {
        JtR[j] += J[i][j] * res[i];
        for (let k = 0; k < np; k++) {
          JtJ[j][k] += J[i][j] * J[i][k];
        }
      }
    }
    // (J^T*J + lambda*diag) * dp = J^T*r
    const A = JtJ.map((row, i) => {
      const r = [...row];
      r[i] += lambda * (r[i] + 1e-10);
      return r;
    });
    // Solve with Gaussian elimination
    const aug = A.map((row, i) => [...row, JtR[i]]);
    for (let col = 0; col < np; col++) {
      let maxRow = col;
      for (let row = col + 1; row < np; row++)
        if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
      [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
      if (Math.abs(aug[col][col]) < 1e-20) continue;
      for (let row = 0; row < np; row++) {
        if (row === col) continue;
        const f = aug[row][col] / aug[col][col];
        for (let j = col; j <= np; j++) aug[row][j] -= f * aug[col][j];
      }
    }
    const dp = aug.map((row, i) => row[np] / (row[i] + 1e-30));

    const newParams = params.map((p, i) => p + dp[i]);
    // Clamp constraints
    newParams[2] = clamp(newParams[2], 0.01, 20); // FWHM > 0
    newParams[3] = clamp(newParams[3], 0, 1); // eta in [0,1]
    if (newParams[0] < 0) newParams[0] = Math.abs(newParams[0]);

    if (chi2(newParams) < chi2(params)) {
      params = newParams;
      lambda *= 0.5;
    } else {
      lambda *= 2;
    }
    if (lambda > 1e10) break;
  }
  // Compute R²
  const yMean = yData.reduce((a, b) => a + b, 0) / n;
  const ssTot = yData.reduce((s, y) => s + (y - yMean) ** 2, 0);
  const ssRes = residuals(params).reduce((s, r) => s + r * r, 0);
  const rSquared = 1 - ssRes / ssTot;

  // Integrated area (trapezoidal rule on the background-subtracted peak)
  const xMin = Math.min(...xData);
  const xMax = Math.max(...xData);
  const steps = 1000;
  const dx = (xMax - xMin) / steps;
  let area = 0;
  for (let i = 0; i <= steps; i++) {
    const x = xMin + i * dx;
    const bg = params[4] + params[5] * x;
    const peak = pseudoVoigt(x, params) - bg;
    area += (i === 0 || i === steps ? 0.5 : 1) * peak * dx;
  }

  return { params, rSquared: clamp(rSquared, 0, 1), area: Math.abs(area) };
}

// ─── XRR Analysis ────────────────────────────────────────────────────────────
// Auto-extracts critical angle (→ density) and Kiessig fringe spacing (→ thickness)
// from a specular reflectivity curve. Angles are treated as incidence angle θ (deg).
function analyzeXRR(data, lambdaA, zOverA) {
  const n = data.length;
  if (n < 20) return null;
  const ang = data.map(d => d.angle);
  const R = data.map(d => d.intensity);
  // Light smoothing (7-pt moving average) for fringe detection only: it suppresses
  // noise spikes that would be mistaken for fringes, but is kept off the sharp
  // critical edge (where smoothing would bias the critical angle high).
  const Rs = smooth(R, 3);

  // Critical angle: where R falls through half of the total-reflection plateau.
  const head = R.slice(0, Math.max(5, Math.floor(n * 0.05)));
  const plateau = median(head);
  const halfMax = plateau / 2;
  const searchEnd = Math.floor(n * 0.5); // the critical edge is always at low angle
  let thetaC = null, edgeIdx = 1;
  for (let i = 1; i < searchEnd; i++) {
    if (R[i] < halfMax && R[i - 1] >= halfMax) {
      const t = (halfMax - R[i - 1]) / (R[i] - R[i - 1]);
      thetaC = ang[i - 1] + t * (ang[i] - ang[i - 1]);
      edgeIdx = i;
      break;
    }
  }
  if (thetaC == null) return null;
  const thetaCrad = thetaC * DEG;

  // Density from the critical angle:  θc² = (re·λ²/π)·n_e
  const neA3 = (Math.PI * thetaCrad * thetaCrad) / (R_E * lambdaA * lambdaA); // e/Å³
  const neCm3 = neA3 * 1e24;                                                 // e/cm³
  const massDensity = neCm3 / (N_A * zOverA);                                // g/cm³

  // Kiessig fringes: local maxima of R above the critical edge.
  const w = 4;
  const fringeAngles = [];
  for (let i = edgeIdx + 3; i < n - w; i++) {
    let isMax = true;
    for (let k = 1; k <= w; k++) {
      if (Rs[i] <= Rs[i - k] || Rs[i] <= Rs[i + k]) { isMax = false; break; }
    }
    if (isMax) fringeAngles.push(ang[i]);
  }

  // Thickness:  q_m = √(sin²θ_m − sin²θc) is linear in fringe order with
  // spacing λ/2t per fringe  →  t = λ / (2·median(Δq)).
  let thickness = null;
  const sin2c = Math.sin(thetaCrad) ** 2;
  const q = fringeAngles
    .map(p => { const s = Math.sin(p * DEG) ** 2 - sin2c; return s > 0 ? Math.sqrt(s) : null; })
    .filter(v => v != null);
  if (q.length >= 2) {
    const dq = [];
    for (let i = 1; i < q.length; i++) dq.push(q[i] - q[i - 1]);
    const medDq = median(dq);
    if (medDq > 0) thickness = lambdaA / (2 * medDq);
  }

  return { thetaC, neA3, massDensity, thickness, nFringes: fringeAngles.length, fringeAngles };
}

// ─── Demo Data Generator ─────────────────────────────────────────────────────
function generateDemoData(mode) {
  const points = [];
  if (mode === "xrr") {
    // Single film: thickness 480 Å, critical angle 0.30° (Cu Kα).
    const nPts = 800, lam = 1.5406, thetaC = 0.30, t = 480, R0 = 1e6;
    const sin2c = Math.sin(thetaC * DEG) ** 2;
    for (let i = 0; i < nPts; i++) {
      const x = 0.05 + (2.95 * i) / (nPts - 1);
      let env, osc = 1;
      if (x <= thetaC) {
        env = R0;
      } else {
        const k = Math.sqrt(x * x - thetaC * thetaC);      // Fresnel small-angle edge
        env = R0 * ((x - k) / (x + k)) ** 2;
        const arg = Math.sin(x * DEG) ** 2 - sin2c;        // Kiessig oscillation
        if (arg > 0) osc = 1 + 0.6 * Math.cos(2 * Math.PI * 2 * t * Math.sqrt(arg) / lam);
      }
      const signal = env * osc + 5;
      const noise = Math.sqrt(Math.max(signal, 1)) * (Math.random() * 2 - 1);
      points.push({ angle: parseFloat(x.toFixed(4)), intensity: Math.max(1, Math.round(signal + noise)) });
    }
    return points;
  }
  const nPts = 500;
  if (mode === "omega") {
    for (let i = 0; i < nPts; i++) {
      const x = -5 + (10 * i) / (nPts - 1);
      const signal = 12000 * Math.exp(-4 * Math.LN2 * (x ** 2) / (1.8 ** 2));
      const noise = Math.sqrt(Math.max(signal, 1)) * (Math.random() * 2 - 1);
      points.push({ angle: parseFloat(x.toFixed(4)), intensity: Math.max(0, Math.round(signal + 50 + noise)) });
    }
  } else {
    for (let i = 0; i < nPts; i++) {
      const x = 34 + (4 * i) / (nPts - 1);
      const signal = 25000 * Math.exp(-4 * Math.LN2 * ((x - 36.04) ** 2) / (0.35 ** 2));
      const noise = Math.sqrt(Math.max(signal, 1)) * (Math.random() * 2 - 1);
      points.push({ angle: parseFloat(x.toFixed(4)), intensity: Math.max(0, Math.round(signal + 120 + noise)) });
    }
  }
  return points;
}

// ─── Parse uploaded file ─────────────────────────────────────────────────────
function parseXRDFile(text) {
  const lines = text.split(/\r?\n/);
  const data = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || trimmed.startsWith("!")) continue;
    const parts = trimmed.split(/[\s,;]+/).map(Number);
    if (parts.length >= 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
      data.push({ angle: parts[0], intensity: parts[1] });
    }
  }
  if (data.length < 5) throw new Error("Could not parse enough data points. Need at least 5 rows of (angle, intensity).");
  return data;
}

// ─── Quality badge ───────────────────────────────────────────────────────────
const QualityBadge = ({ fwhm }) => {
  let color, label;
  if (fwhm < 1) { color = C.good; label = "Excellent"; }
  else if (fwhm < 2) { color = C.warn; label = "Acceptable"; }
  else { color = C.bad; label = "Poor — grain boundaries too large"; }
  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "5px 12px", borderRadius: 6, border: `1px solid ${color}`, background: "#fff" }}>
      <div style={{ width: 8, height: 8, borderRadius: "50%", background: color }} />
      <span style={{ color, fontWeight: 600, fontSize: 13 }}>{label}</span>
    </div>
  );
};

// ─── Main Component ──────────────────────────────────────────────────────────
export default function XRDDashboard() {
  const [scanMode, setScanMode] = useState("omega");
  const [rawData, setRawData] = useState(null);
  const [fileName, setFileName] = useState("");
  const [fitRange, setFitRange] = useState([null, null]);
  const [fitResult, setFitResult] = useState(null);
  const [fitCurve, setFitCurve] = useState([]);
  const [logScale, setLogScale] = useState(false);
  const [sampleLog, setSampleLog] = useState([]);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [wavelength, setWavelength] = useState(1.5406);
  const [zOverA, setZOverA] = useState(0.5);
  const [xrrResult, setXrrResult] = useState(null);
  const fileInputRef = useRef();

  const isXRR = scanMode === "xrr";

  // Compute default fit range when data loads
  useEffect(() => {
    if (rawData && rawData.length > 0) {
      const angles = rawData.map(d => d.angle);
      setFitRange([Math.min(...angles), Math.max(...angles)]);
    }
  }, [rawData]);

  // Run analysis whenever data, range, mode, or XRR inputs change
  useEffect(() => {
    if (!rawData) {
      setFitResult(null); setFitCurve([]); setXrrResult(null);
      return;
    }
    if (isXRR) {
      setFitResult(null); setFitCurve([]);
      setXrrResult(analyzeXRR(rawData, wavelength, zOverA));
      return;
    }
    setXrrResult(null);
    if (!fitRange[0] || !fitRange[1] || fitRange[0] >= fitRange[1]) {
      setFitResult(null);
      setFitCurve([]);
      return;
    }
    try {
      const subset = rawData.filter(d => d.angle >= fitRange[0] && d.angle <= fitRange[1]);
      if (subset.length < 5) { setFitResult(null); setFitCurve([]); return; }
      const xArr = subset.map(d => d.angle);
      const yArr = subset.map(d => d.intensity);
      const maxIdx = yArr.indexOf(Math.max(...yArr));
      const x0Guess = xArr[maxIdx];
      const I0Guess = yArr[maxIdx];
      const bgGuess = Math.min(...yArr);
      const initial = [I0Guess - bgGuess, x0Guess, 0.5, 0.5, bgGuess, 0];
      const result = fitPseudoVoigt(xArr, yArr, initial);
      const [I0, center, fwhm, eta, a, b] = result.params;
      setFitResult({ center, fwhm: Math.abs(fwhm), intensity: I0, eta, rSquared: result.rSquared, area: result.area, bgA: a, bgB: b });

      // Generate smooth fit curve
      const xMin = Math.min(...xArr);
      const xMax = Math.max(...xArr);
      const curve = [];
      for (let i = 0; i <= 300; i++) {
        const x = xMin + (xMax - xMin) * (i / 300);
        curve.push({ angle: x, fitted: pseudoVoigt(x, result.params) });
      }
      setFitCurve(curve);
    } catch {
      setFitResult(null);
      setFitCurve([]);
    }
  }, [rawData, fitRange, isXRR, wavelength, zOverA]);

  const handleFile = useCallback((file) => {
    setError("");
    if (!file) return;
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = parseXRDFile(e.target.result);
        setRawData(data);
      } catch (err) {
        setError(err.message);
        setRawData(null);
      }
    };
    reader.readAsText(file);
  }, []);

  const loadDemo = useCallback(() => {
    setError("");
    const names = { omega: "demo_rocking_curve.xy", theta2theta: "demo_theta2theta.xy", xrr: "demo_xrr.xy" };
    setFileName(names[scanMode]);
    setRawData(generateDemoData(scanMode));
  }, [scanMode]);

  const addToLog = useCallback(() => {
    if (!fitResult) return;
    setSampleLog(prev => [...prev, {
      id: Date.now(),
      name: fileName || "Untitled",
      scanType: scanMode === "omega" ? "Rocking Curve" : "θ-2θ",
      center: fitResult.center.toFixed(4),
      fwhm: fitResult.fwhm.toFixed(4),
      intensity: fitResult.intensity.toFixed(1),
      quality: scanMode === "omega" ? (fitResult.fwhm < 1 ? "Excellent" : fitResult.fwhm < 2 ? "Acceptable" : "Poor") : "—",
      notes: ""
    }]);
  }, [fitResult, fileName, scanMode]);

  const exportCSV = useCallback(() => {
    if (sampleLog.length === 0) return;
    const header = "Sample Name,Scan Type,Peak Center,FWHM,Intensity,Quality,Notes\n";
    const rows = sampleLog.map(r => `"${r.name}","${r.scanType}",${r.center},${r.fwhm},${r.intensity},"${r.quality}","${r.notes}"`).join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "xrd_sample_log.csv"; a.click();
    URL.revokeObjectURL(url);
  }, [sampleLog]);

  const xLabel = isXRR ? "Incidence angle θ (°)" : scanMode === "omega" ? "ω (°)" : "2θ (°)";
  const yLabel = isXRR ? "Reflectivity (counts)" : "Intensity (counts)";
  const fmtCounts = (v) => v >= 1e6 ? (v / 1e6) + "M" : v >= 1e3 ? (v / 1e3) + "k" : v;

  // XRR plot data (positive-floored for log axis) + fringe markers
  const xrrData = isXRR && rawData ? rawData.map(d => ({ angle: d.angle, R: Math.max(d.intensity, 1) })) : [];
  const fringeMarkers = isXRR && xrrResult ? xrrResult.fringeAngles.map(a => {
    const pt = rawData.reduce((best, d) => Math.abs(d.angle - a) < Math.abs(best.angle - a) ? d : best, rawData[0]);
    return { angle: a, R: Math.max(pt.intensity, 1) };
  }) : [];

  const styles = {
    root: { fontFamily: SANS, background: C.bg, color: C.text, minHeight: "100vh", padding: 0 },
    header: { background: C.panel, borderBottom: `1px solid ${C.border}`, padding: "14px 28px", display: "flex", alignItems: "center", justifyContent: "space-between", position: "sticky", top: 0, zIndex: 50 },
    title: { fontSize: 17, fontWeight: 700, letterSpacing: 0.2, color: C.text, margin: 0 },
    subtitle: { fontSize: 12, color: C.textMuted, marginTop: 2, fontWeight: 400 },
    tab: (active) => ({ padding: "6px 2px", margin: "0 12px", border: "none", borderBottom: `2px solid ${active ? C.accent : "transparent"}`, background: "transparent", color: active ? C.accent : C.textMuted, cursor: "pointer", fontSize: 13, fontWeight: active ? 600 : 500, letterSpacing: 0.2 }),
    card: { background: C.panel, border: `1px solid ${C.border}`, borderRadius: 8, padding: 20 },
    btn: (variant) => ({
      padding: "8px 16px", borderRadius: 6,
      border: variant === "primary" ? `1px solid ${C.accent}` : `1px solid ${C.borderStrong}`,
      background: variant === "primary" ? C.accent : C.panel,
      color: variant === "primary" ? "#fff" : C.text, cursor: "pointer",
      fontSize: 13, fontWeight: 600, letterSpacing: 0.2
    }),
    dropzone: { border: `1.5px dashed ${dragOver ? C.accent : C.borderStrong}`, borderRadius: 8, padding: "26px 16px", textAlign: "center", cursor: "pointer", background: dragOver ? C.accentSoft : "transparent", transition: "all 0.2s" },
    metricCard: { background: "#fafbfc", borderRadius: 6, padding: "10px 14px", border: `1px solid ${C.border}` },
    metricLabel: { fontSize: 11, color: C.textMuted, fontWeight: 500, marginBottom: 3, letterSpacing: 0.2 },
    metricValue: { fontSize: 19, fontWeight: 600, color: C.text, fontVariantNumeric: "tabular-nums" },
    input: { background: C.panel, border: `1px solid ${C.borderStrong}`, borderRadius: 6, padding: "6px 9px", color: C.text, fontSize: 13, fontFamily: "inherit", width: 90, outline: "none" },
    sectionLabel: { fontSize: 11, color: C.textMuted, letterSpacing: 0.6, textTransform: "uppercase", fontWeight: 600, marginBottom: 14 },
  };

  return (
    <div style={styles.root}>
      {/* Header */}
      <div style={styles.header}>
        <div>
          <div style={styles.title}>XRD Peak Analyzer</div>
          <div style={styles.subtitle}>Thin-Film X-ray Characterization Suite</div>
        </div>
        <div style={{ display: "flex", alignItems: "center" }}>
          <button style={styles.tab(scanMode === "theta2theta")} onClick={() => setScanMode("theta2theta")}>θ–2θ Scan</button>
          <button style={styles.tab(scanMode === "omega")} onClick={() => setScanMode("omega")}>ω Rocking Curve</button>
          <button style={styles.tab(isXRR)} onClick={() => setScanMode("xrr")}>XRR Reflectivity</button>
        </div>
      </div>

      <div style={{ padding: "24px 28px", maxWidth: 1400, margin: "0 auto" }}>
        {/* Upload + Controls Row */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 16, marginBottom: 20 }}>
          <div style={{ ...styles.card }}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]); }}
          >
            <div style={styles.dropzone} onClick={() => fileInputRef.current?.click()}>
              <input ref={fileInputRef} type="file" accept=".xy,.csv,.txt,.dat" style={{ display: "none" }}
                onChange={(e) => handleFile(e.target.files[0])} />
              <div style={{ fontSize: 14, color: C.text, fontWeight: 500 }}>
                {fileName ? `Loaded: ${fileName}` : "Drop a .xy / .csv file here, or click to browse"}
              </div>
              <div style={{ fontSize: 12, color: C.textFaint, marginTop: 6 }}>Two-column format: angle &nbsp; intensity</div>
            </div>
          </div>
          <div style={{ ...styles.card, display: "flex", flexDirection: "column", gap: 12, justifyContent: "center", minWidth: 220 }}>
            <button style={styles.btn("primary")} onClick={loadDemo}>Load demo data</button>
            {!isXRR && rawData && fitResult && <button style={styles.btn()} onClick={addToLog}>Add to sample log</button>}
            {scanMode === "theta2theta" && (
              <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, color: C.textMuted, cursor: "pointer" }}>
                <input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)}
                  style={{ accentColor: C.accent }} />
                Log scale Y
              </label>
            )}
            {isXRR && (
              <>
                <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12, color: C.textMuted }}>
                  <span>Wavelength λ (Å)</span>
                  <input type="number" step="0.0001" value={wavelength} style={{ ...styles.input, width: "100%" }}
                    onChange={(e) => setWavelength(parseFloat(e.target.value) || 0)} />
                </label>
                <label style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 12, color: C.textMuted }}>
                  <span>Z/A ratio</span>
                  <input type="number" step="0.001" value={zOverA} style={{ ...styles.input, width: "100%" }}
                    onChange={(e) => setZOverA(parseFloat(e.target.value) || 0)} />
                </label>
              </>
            )}
          </div>
        </div>

        {error && (
          <div style={{ background: "#fdf2f2", border: `1px solid ${C.bad}`, borderRadius: 8, padding: "12px 16px", marginBottom: 16, color: C.bad, fontSize: 13 }}>
            {error}
          </div>
        )}

        {/* Fit Range Controls (peak modes only) */}
        {!isXRR && rawData && (
          <div style={{ ...styles.card, marginBottom: 20, display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
            <span style={{ fontSize: 11, color: C.textMuted, letterSpacing: 0.6, textTransform: "uppercase", fontWeight: 600 }}>Fit Range</span>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input type="number" step="0.1" value={fitRange[0] ?? ""} style={styles.input}
                onChange={(e) => setFitRange([parseFloat(e.target.value) || 0, fitRange[1]])} />
              <span style={{ color: C.textFaint }}>→</span>
              <input type="number" step="0.1" value={fitRange[1] ?? ""} style={styles.input}
                onChange={(e) => setFitRange([fitRange[0], parseFloat(e.target.value) || 0])} />
              <span style={{ fontSize: 12, color: C.textMuted }}>degrees</span>
            </div>
            {fitResult && (
              <span style={{ marginLeft: "auto", fontSize: 12, color: C.accent, fontWeight: 600 }}>
                η = {fitResult.eta?.toFixed(3)} ({fitResult.eta < 0.3 ? "Gaussian-like" : fitResult.eta > 0.7 ? "Lorentzian-like" : "Mixed"})
              </span>
            )}
          </div>
        )}

        {/* Main Content: Plot + Results */}
        {rawData && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 20, marginBottom: 20 }}>
            {/* Plot */}
            <div style={{ ...styles.card, padding: 16 }}>
              <div style={{ ...styles.sectionLabel, marginBottom: 12 }}>
                {isXRR ? "X-ray Reflectivity" : scanMode === "omega" ? "Rocking Curve Profile" : "θ–2θ Diffraction Profile"}
              </div>
              <ResponsiveContainer width="100%" height={420}>
                {isXRR ? (
                  <ComposedChart data={xrrData} margin={{ top: 10, right: 20, bottom: 40, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.grid} />
                    <XAxis dataKey="angle" type="number" domain={["dataMin", "dataMax"]}
                      tick={{ fill: C.textMuted, fontSize: 11 }}
                      label={{ value: xLabel, position: "bottom", offset: 20, fill: C.textMuted, fontSize: 12 }}
                      stroke={C.borderStrong}
                    />
                    <YAxis scale="log" domain={[1, "auto"]} allowDataOverflow
                      tick={{ fill: C.textMuted, fontSize: 11 }}
                      label={{ value: yLabel, angle: -90, position: "insideLeft", offset: -5, fill: C.textMuted, fontSize: 12 }}
                      stroke={C.borderStrong}
                      tickFormatter={fmtCounts}
                    />
                    <Tooltip
                      contentStyle={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12, color: C.text }}
                      formatter={(value) => [typeof value === "number" ? value.toFixed(0) : value, "R"]}
                      labelFormatter={(v) => `θ: ${parseFloat(v).toFixed(3)}°`}
                    />
                    <Line dataKey="R" dot={false} stroke="#334155" strokeWidth={1.3} isAnimationActive={false} name="Reflectivity" />
                    {xrrResult?.thetaC != null && (
                      <ReferenceLine x={xrrResult.thetaC} stroke={C.accent} strokeDasharray="5 4" strokeWidth={1.3}
                        label={{ value: "θc", position: "top", fill: C.accent, fontSize: 11 }} />
                    )}
                    {fringeMarkers.length > 0 && (
                      <Scatter data={fringeMarkers} dataKey="R" fill={C.marker} isAnimationActive={false} name="Kiessig fringe" />
                    )}
                  </ComposedChart>
                ) : (
                  <ComposedChart data={rawData} margin={{ top: 10, right: 20, bottom: 40, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.grid} />
                    {fitRange[0] != null && fitRange[1] != null && (
                      <ReferenceArea x1={fitRange[0]} x2={fitRange[1]} fill="rgba(37,99,235,0.05)" stroke="rgba(37,99,235,0.2)" strokeDasharray="4 4" />
                    )}
                    <XAxis dataKey="angle" type="number" domain={["dataMin", "dataMax"]}
                      tick={{ fill: C.textMuted, fontSize: 11 }}
                      label={{ value: xLabel, position: "bottom", offset: 20, fill: C.textMuted, fontSize: 12 }}
                      stroke={C.borderStrong}
                    />
                    <YAxis
                      scale={logScale && scanMode === "theta2theta" ? "log" : "auto"}
                      domain={logScale && scanMode === "theta2theta" ? [1, "auto"] : [0, "auto"]}
                      tick={{ fill: C.textMuted, fontSize: 11 }}
                      label={{ value: yLabel, angle: -90, position: "insideLeft", offset: -5, fill: C.textMuted, fontSize: 12 }}
                      stroke={C.borderStrong}
                      tickFormatter={fmtCounts}
                    />
                    <Tooltip
                      contentStyle={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12, color: C.text }}
                      formatter={(value, name) => [typeof value === "number" ? value.toFixed(2) : value, name === "intensity" ? "Raw" : "Fit"]}
                      labelFormatter={(v) => `${xLabel.replace(" (°)", "")}: ${parseFloat(v).toFixed(3)}°`}
                    />
                    <Scatter name="intensity" dataKey="intensity" fill="rgba(71,85,105,0.55)" isAnimationActive={false} />
                    {fitCurve.length > 0 && (
                      <Line data={fitCurve} dataKey="fitted" xAxisId={0} dot={false} stroke={C.accent} strokeWidth={2}
                        isAnimationActive={false} name="Fit" connectNulls />
                    )}
                    {fitResult && (
                      <ReferenceLine x={fitResult.center} stroke={C.marker} strokeDasharray="6 4" strokeWidth={1.5} />
                    )}
                  </ComposedChart>
                )}
              </ResponsiveContainer>
              <div style={{ display: "flex", gap: 20, justifyContent: "center", marginTop: 8, flexWrap: "wrap" }}>
                {isXRR ? (
                  <>
                    <span style={{ fontSize: 12, color: C.textMuted, display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ display: "inline-block", width: 16, height: 2, background: "#334155" }} /> Reflectivity
                    </span>
                    <span style={{ fontSize: 12, color: C.textMuted, display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ display: "inline-block", width: 16, height: 0, borderTop: `2px dashed ${C.accent}` }} /> Critical angle
                    </span>
                    <span style={{ fontSize: 12, color: C.textMuted, display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: C.marker }} /> Kiessig fringe
                    </span>
                  </>
                ) : (
                  <>
                    <span style={{ fontSize: 12, color: C.textMuted, display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: "rgba(71,85,105,0.55)" }} /> Raw data
                    </span>
                    <span style={{ fontSize: 12, color: C.textMuted, display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ display: "inline-block", width: 16, height: 2, background: C.accent }} /> Pseudo-Voigt fit
                    </span>
                    <span style={{ fontSize: 12, color: C.textMuted, display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ display: "inline-block", width: 16, height: 0, borderTop: `2px dashed ${C.marker}` }} /> Peak center
                    </span>
                  </>
                )}
              </div>
            </div>

            {/* Results Panel */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {/* XRR Results */}
              {isXRR ? (
                <div style={{ ...styles.card }}>
                  <div style={styles.sectionLabel}>Reflectivity Results</div>
                  {xrrResult ? (
                    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                      <div style={styles.metricCard}>
                        <div style={styles.metricLabel}>Critical Angle θc</div>
                        <div style={styles.metricValue}>{xrrResult.thetaC.toFixed(4)}°</div>
                      </div>
                      <div style={styles.metricCard}>
                        <div style={styles.metricLabel}>Film Thickness</div>
                        <div style={styles.metricValue}>
                          {xrrResult.thickness != null
                            ? `${xrrResult.thickness.toFixed(1)} Å`
                            : "Not detected"}
                        </div>
                        {xrrResult.thickness != null && (
                          <div style={{ fontSize: 12, color: C.textMuted, marginTop: 2 }}>
                            {(xrrResult.thickness / 10).toFixed(2)} nm · {xrrResult.nFringes} fringes
                          </div>
                        )}
                      </div>
                      <div style={styles.metricCard}>
                        <div style={styles.metricLabel}>Electron Density</div>
                        <div style={styles.metricValue}>{xrrResult.neA3.toFixed(4)}</div>
                        <div style={{ fontSize: 12, color: C.textMuted, marginTop: 2 }}>e⁻ / Å³</div>
                      </div>
                      <div style={styles.metricCard}>
                        <div style={styles.metricLabel}>Mass Density</div>
                        <div style={styles.metricValue}>{xrrResult.massDensity.toFixed(3)}</div>
                        <div style={{ fontSize: 12, color: C.textMuted, marginTop: 2 }}>g / cm³ (Z/A = {zOverA})</div>
                      </div>
                    </div>
                  ) : (
                    <div style={{ color: C.textFaint, fontSize: 13, textAlign: "center", padding: 20 }}>
                      No critical edge found. Scan should start below θc.
                    </div>
                  )}
                </div>
              ) : (
                <>
                  <div style={{ ...styles.card }}>
                    <div style={styles.sectionLabel}>Fit Results</div>
                    {fitResult ? (
                      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                        <div style={styles.metricCard}>
                          <div style={styles.metricLabel}>Peak Center</div>
                          <div style={styles.metricValue}>{fitResult.center.toFixed(4)}°</div>
                        </div>
                        <div style={styles.metricCard}>
                          <div style={styles.metricLabel}>FWHM</div>
                          <div style={{ ...styles.metricValue, color: scanMode === "omega" ? (fitResult.fwhm < 1 ? C.good : fitResult.fwhm < 2 ? C.warn : C.bad) : C.text }}>
                            {fitResult.fwhm.toFixed(4)}°
                          </div>
                        </div>
                        <div style={styles.metricCard}>
                          <div style={styles.metricLabel}>Peak Intensity</div>
                          <div style={styles.metricValue}>{fitResult.intensity.toFixed(1)}</div>
                        </div>
                        <div style={styles.metricCard}>
                          <div style={styles.metricLabel}>Integrated Area</div>
                          <div style={styles.metricValue}>{fitResult.area.toFixed(1)}</div>
                        </div>
                        <div style={styles.metricCard}>
                          <div style={styles.metricLabel}>R² Goodness of Fit</div>
                          <div style={{ ...styles.metricValue, color: fitResult.rSquared > 0.99 ? C.good : fitResult.rSquared > 0.95 ? C.warn : C.bad }}>
                            {fitResult.rSquared.toFixed(6)}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div style={{ color: C.textFaint, fontSize: 13, textAlign: "center", padding: 20 }}>
                        Adjust fit range to see results
                      </div>
                    )}
                  </div>

                  {/* Quality Badge (Omega mode only) */}
                  {scanMode === "omega" && fitResult && (
                    <div style={{ ...styles.card }}>
                      <div style={styles.sectionLabel}>Crystal Quality</div>
                      <QualityBadge fwhm={fitResult.fwhm} />
                      <div style={{ fontSize: 12, color: C.textMuted, marginTop: 12, lineHeight: 1.6 }}>
                        Target: FWHM &lt; 2° for low-angle grain boundary requirement
                      </div>
                    </div>
                  )}

                  {/* Fit Parameters */}
                  {fitResult && (
                    <div style={{ ...styles.card }}>
                      <div style={styles.sectionLabel}>Fit Parameters</div>
                      <div style={{ fontSize: 13, color: C.textMuted, lineHeight: 2 }}>
                        <div>η (mixing) = <span style={{ color: C.text, fontWeight: 600 }}>{fitResult.eta?.toFixed(4)}</span></div>
                        <div>Background = <span style={{ color: C.text, fontWeight: 600 }}>{fitResult.bgA?.toFixed(2)} {fitResult.bgB >= 0 ? "+" : "−"} {Math.abs(fitResult.bgB).toFixed(4)}·x</span></div>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}

        {/* Empty State */}
        {!rawData && !error && (
          <div style={{ ...styles.card, textAlign: "center", padding: "60px 20px", marginBottom: 20 }}>
            <div style={{ fontSize: 14, color: C.textMuted, maxWidth: 440, margin: "0 auto", lineHeight: 1.8 }}>
              Upload an X-ray data file to begin analysis, or click <strong style={{ color: C.accent }}>Load demo data</strong> to explore with synthetic {isXRR ? "reflectivity" : "AlN (0002)"} data.
            </div>
          </div>
        )}

        {/* Sample Log Table */}
        {sampleLog.length > 0 && (
          <div style={{ ...styles.card }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <div style={styles.sectionLabel}>Sample Log ({sampleLog.length})</div>
              <button style={styles.btn("primary")} onClick={exportCSV}>Export CSV</button>
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>
                    {["Sample", "Scan", "Center (°)", "FWHM (°)", "Intensity", "Quality", "Notes"].map(h => (
                      <th key={h} style={{ textAlign: "left", padding: "8px 12px", borderBottom: `1px solid ${C.border}`, color: C.textMuted, fontSize: 11, letterSpacing: 0.3, textTransform: "uppercase", fontWeight: 600 }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sampleLog.map((row, idx) => (
                    <tr key={row.id} style={{ background: idx % 2 === 0 ? "transparent" : "#fafbfc" }}>
                      <td style={{ padding: "8px 12px", color: C.text }}>{row.name}</td>
                      <td style={{ padding: "8px 12px", color: C.textMuted }}>{row.scanType}</td>
                      <td style={{ padding: "8px 12px", color: C.text, fontVariantNumeric: "tabular-nums" }}>{row.center}</td>
                      <td style={{ padding: "8px 12px", color: C.text, fontVariantNumeric: "tabular-nums" }}>{row.fwhm}</td>
                      <td style={{ padding: "8px 12px", color: C.text, fontVariantNumeric: "tabular-nums" }}>{row.intensity}</td>
                      <td style={{ padding: "8px 12px" }}>
                        <span style={{
                          color: row.quality === "Excellent" ? C.good : row.quality === "Acceptable" ? C.warn : row.quality === "Poor" ? C.bad : C.textFaint,
                          fontWeight: 600
                        }}>{row.quality}</span>
                      </td>
                      <td style={{ padding: "8px 12px" }}>
                        <input type="text" value={row.notes} placeholder="Add notes…"
                          style={{ ...styles.input, width: "100%", background: "transparent", border: `1px solid ${C.border}` }}
                          onChange={(e) => {
                            const val = e.target.value;
                            setSampleLog(prev => prev.map((r, i) => i === idx ? { ...r, notes: val } : r));
                          }} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Footer */}
        <div style={{ textAlign: "center", padding: "24px 0 12px", borderTop: `1px solid ${C.border}`, marginTop: 8 }}>
          <div style={{ fontSize: 12, color: C.textMuted, marginBottom: 6 }}>
            XRD Peak Analyzer · Pseudo-Voigt fitting · XRR density &amp; thickness
          </div>
          <div style={{ fontSize: 12, color: C.textFaint }}>
            © 2026 Zhaosen Qu
          </div>
        </div>
      </div>
    </div>
  );
}
