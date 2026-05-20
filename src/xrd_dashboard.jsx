import { useState, useEffect, useCallback, useRef } from "react";
import {
  ScatterChart, Scatter, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceArea, ReferenceLine, ComposedChart,
  Area
} from "recharts";

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

  // Integrated area (numerical)
  const xMin = Math.min(...xData);
  const xMax = Math.max(...xData);
  const steps = 1000;
  const dx = (xMax - xMin) / steps;
  let area = 0;
  for (let i = 0; i <= steps; i++) {
    const x = xMin + i * dx;
    const bg = params[4] + params[5] * x;
    area += (pseudoVoigt(x, params) - bg) * dx;
  }

  return { params, rSquared: clamp(rSquared, 0, 1), area: Math.abs(area) };
}

// ─── Demo Data Generator ─────────────────────────────────────────────────────
function generateDemoData(mode) {
  const points = [];
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
  let color, label, bg;
  if (fwhm < 1) { color = "#34d399"; label = "Excellent"; bg = "rgba(52,211,153,0.15)"; }
  else if (fwhm < 2) { color = "#fbbf24"; label = "Acceptable"; bg = "rgba(251,191,36,0.15)"; }
  else { color = "#f87171"; label = "Poor — grain boundaries too large"; bg = "rgba(248,113,113,0.15)"; }
  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "6px 14px", borderRadius: 8, border: `1.5px solid ${color}`, background: bg }}>
      <div style={{ width: 10, height: 10, borderRadius: "50%", background: color, boxShadow: `0 0 8px ${color}` }} />
      <span style={{ color, fontWeight: 700, fontSize: 13, letterSpacing: 0.5 }}>{label}</span>
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
  const fileInputRef = useRef();

  // Compute default fit range when data loads
  useEffect(() => {
    if (rawData && rawData.length > 0) {
      const angles = rawData.map(d => d.angle);
      setFitRange([Math.min(...angles), Math.max(...angles)]);
    }
  }, [rawData]);

  // Run fit whenever data or range changes
  useEffect(() => {
    if (!rawData || !fitRange[0] || !fitRange[1] || fitRange[0] >= fitRange[1]) {
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
  }, [rawData, fitRange]);

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
    setFileName(scanMode === "omega" ? "demo_rocking_curve.xy" : "demo_theta2theta.xy");
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

  const xLabel = scanMode === "omega" ? "ω (°)" : "2θ (°)";
  const yLabel = "Intensity (counts)";

  // Merge raw + fit for the composed chart
  const chartData = rawData ? rawData.map(d => {
    const fitPt = fitCurve.length > 0 ? fitCurve.reduce((best, c) => Math.abs(c.angle - d.angle) < Math.abs(best.angle - d.angle) ? c : best, fitCurve[0]) : null;
    return { angle: d.angle, intensity: d.intensity, fitted: fitPt && Math.abs(fitPt.angle - d.angle) < 0.05 ? fitPt.fitted : undefined };
  }) : [];

  const styles = {
    root: { fontFamily: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace", background: "linear-gradient(170deg, #0b1120 0%, #0f172a 40%, #131c33 100%)", color: "#e2e8f0", minHeight: "100vh", padding: "0" },
    header: { background: "rgba(15,23,42,0.9)", borderBottom: "1px solid rgba(56,189,248,0.15)", padding: "16px 24px", display: "flex", alignItems: "center", justifyContent: "space-between", backdropFilter: "blur(12px)", position: "sticky", top: 0, zIndex: 50 },
    title: { fontSize: 18, fontWeight: 800, letterSpacing: 1.5, color: "#38bdf8", textTransform: "uppercase", margin: 0 },
    subtitle: { fontSize: 11, color: "#64748b", letterSpacing: 2, fontWeight: 500, marginTop: 2 },
    tab: (active) => ({ padding: "8px 20px", border: "1px solid " + (active ? "#38bdf8" : "rgba(100,116,139,0.3)"), borderRadius: 6, background: active ? "rgba(56,189,248,0.12)" : "transparent", color: active ? "#38bdf8" : "#94a3b8", cursor: "pointer", fontSize: 12, fontWeight: 700, letterSpacing: 0.8, transition: "all 0.2s" }),
    card: { background: "rgba(15,23,42,0.6)", border: "1px solid rgba(56,189,248,0.08)", borderRadius: 12, padding: 20, backdropFilter: "blur(8px)" },
    btn: (variant) => ({
      padding: "8px 18px", borderRadius: 6, border: variant === "primary" ? "1px solid #38bdf8" : "1px solid rgba(100,116,139,0.3)",
      background: variant === "primary" ? "rgba(56,189,248,0.15)" : "transparent",
      color: variant === "primary" ? "#38bdf8" : "#94a3b8", cursor: "pointer",
      fontSize: 12, fontWeight: 700, letterSpacing: 0.5, transition: "all 0.2s"
    }),
    dropzone: { border: `2px dashed ${dragOver ? "#38bdf8" : "rgba(100,116,139,0.3)"}`, borderRadius: 12, padding: "28px 16px", textAlign: "center", cursor: "pointer", background: dragOver ? "rgba(56,189,248,0.05)" : "transparent", transition: "all 0.3s" },
    metricCard: { background: "rgba(30,41,59,0.5)", borderRadius: 8, padding: "12px 16px", border: "1px solid rgba(56,189,248,0.06)" },
    metricLabel: { fontSize: 10, color: "#64748b", letterSpacing: 1.5, textTransform: "uppercase", fontWeight: 600, marginBottom: 4 },
    metricValue: { fontSize: 20, fontWeight: 800, color: "#f1f5f9" },
    input: { background: "rgba(15,23,42,0.8)", border: "1px solid rgba(100,116,139,0.3)", borderRadius: 6, padding: "6px 10px", color: "#e2e8f0", fontSize: 13, fontFamily: "inherit", width: 90, outline: "none" }
  };

  return (
    <div style={styles.root}>
      {/* Header */}
      <div style={styles.header}>
        <div>
          <div style={styles.title}>⬡ XRD Peak Analyzer</div>
          <div style={styles.subtitle}>AlN Thin Film Characterization Suite</div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <div style={styles.tab(scanMode === "theta2theta")} onClick={() => setScanMode("theta2theta")}>θ–2θ SCAN</div>
          <div style={styles.tab(scanMode === "omega")} onClick={() => setScanMode("omega")}>ω ROCKING CURVE</div>
        </div>
      </div>

      <div style={{ padding: "20px 24px", maxWidth: 1400, margin: "0 auto" }}>
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
              <div style={{ fontSize: 28, marginBottom: 8 }}>📂</div>
              <div style={{ fontSize: 13, color: "#94a3b8", fontWeight: 600 }}>
                {fileName ? `Loaded: ${fileName}` : "Drop .xy / .csv file here or click to browse"}
              </div>
              <div style={{ fontSize: 11, color: "#475569", marginTop: 6 }}>Two-column format: angle  intensity</div>
            </div>
          </div>
          <div style={{ ...styles.card, display: "flex", flexDirection: "column", gap: 10, justifyContent: "center", minWidth: 200 }}>
            <button style={styles.btn("primary")} onClick={loadDemo}>▶ Load Demo Data</button>
            {rawData && fitResult && <button style={styles.btn()} onClick={addToLog}>＋ Add to Sample Log</button>}
            {scanMode === "theta2theta" && (
              <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "#94a3b8", cursor: "pointer" }}>
                <input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)}
                  style={{ accentColor: "#38bdf8" }} />
                Log scale Y
              </label>
            )}
          </div>
        </div>

        {error && (
          <div style={{ background: "rgba(248,113,113,0.1)", border: "1px solid rgba(248,113,113,0.3)", borderRadius: 8, padding: "12px 16px", marginBottom: 16, color: "#f87171", fontSize: 13 }}>
            ⚠ {error}
          </div>
        )}

        {/* Fit Range Controls */}
        {rawData && (
          <div style={{ ...styles.card, marginBottom: 20, display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
            <span style={{ fontSize: 11, color: "#64748b", letterSpacing: 1, textTransform: "uppercase", fontWeight: 700 }}>Fit Range</span>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <input type="number" step="0.1" value={fitRange[0] ?? ""} style={styles.input}
                onChange={(e) => setFitRange([parseFloat(e.target.value) || 0, fitRange[1]])} />
              <span style={{ color: "#475569" }}>→</span>
              <input type="number" step="0.1" value={fitRange[1] ?? ""} style={styles.input}
                onChange={(e) => setFitRange([fitRange[0], parseFloat(e.target.value) || 0])} />
              <span style={{ fontSize: 11, color: "#64748b" }}>degrees</span>
            </div>
            {fitResult && (
              <span style={{ marginLeft: "auto", fontSize: 11, color: "#38bdf8", fontWeight: 600 }}>
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
              <div style={{ fontSize: 11, color: "#64748b", letterSpacing: 1, textTransform: "uppercase", fontWeight: 700, marginBottom: 12 }}>
                {scanMode === "omega" ? "Rocking Curve Profile" : "θ–2θ Diffraction Profile"}
              </div>
              <ResponsiveContainer width="100%" height={420}>
                <ComposedChart data={rawData} margin={{ top: 10, right: 20, bottom: 40, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.12)" />
                  {fitRange[0] != null && fitRange[1] != null && (
                    <ReferenceArea x1={fitRange[0]} x2={fitRange[1]} fill="rgba(56,189,248,0.04)" stroke="rgba(56,189,248,0.15)" strokeDasharray="4 4" />
                  )}
                  <XAxis dataKey="angle" type="number" domain={["dataMin", "dataMax"]}
                    tick={{ fill: "#94a3b8", fontSize: 11, fontFamily: "inherit" }}
                    label={{ value: xLabel, position: "bottom", offset: 20, fill: "#94a3b8", fontSize: 12 }}
                    stroke="rgba(100,116,139,0.2)"
                  />
                  <YAxis
                    scale={logScale && scanMode === "theta2theta" ? "log" : "auto"}
                    domain={logScale && scanMode === "theta2theta" ? [1, "auto"] : [0, "auto"]}
                    tick={{ fill: "#94a3b8", fontSize: 11, fontFamily: "inherit" }}
                    label={{ value: yLabel, angle: -90, position: "insideLeft", offset: -5, fill: "#94a3b8", fontSize: 12 }}
                    stroke="rgba(100,116,139,0.2)"
                    tickFormatter={(v) => v >= 1000 ? (v / 1000).toFixed(1) + "k" : v}
                  />
                  <Tooltip
                    contentStyle={{ background: "#1e293b", border: "1px solid rgba(56,189,248,0.2)", borderRadius: 8, fontSize: 12, fontFamily: "inherit", color: "#e2e8f0" }}
                    formatter={(value, name) => [typeof value === "number" ? value.toFixed(2) : value, name === "intensity" ? "Raw" : "Fit"]}
                    labelFormatter={(v) => `${xLabel.replace(" (°)", "")}: ${parseFloat(v).toFixed(3)}°`}
                  />
                  <Scatter name="intensity" dataKey="intensity" fill="rgba(148,163,184,0.5)" r={1.8} isAnimationActive={false} />
                  {fitCurve.length > 0 && (
                    <Line data={fitCurve} dataKey="fitted" xAxisId={0} dot={false} stroke="#38bdf8" strokeWidth={2.5}
                      isAnimationActive={false} name="Fit" connectNulls />
                  )}
                  {fitResult && (
                    <ReferenceLine x={fitResult.center} stroke="#f59e0b" strokeDasharray="6 4" strokeWidth={1.5} />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
              <div style={{ display: "flex", gap: 20, justifyContent: "center", marginTop: 8 }}>
                <span style={{ fontSize: 11, color: "#94a3b8", display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: "rgba(148,163,184,0.5)" }} /> Raw data
                </span>
                <span style={{ fontSize: 11, color: "#94a3b8", display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ display: "inline-block", width: 16, height: 3, background: "#38bdf8", borderRadius: 2 }} /> Pseudo-Voigt fit
                </span>
                <span style={{ fontSize: 11, color: "#94a3b8", display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ display: "inline-block", width: 16, height: 0, borderTop: "2px dashed #f59e0b" }} /> Peak center
                </span>
              </div>
            </div>

            {/* Results Panel */}
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div style={{ ...styles.card }}>
                <div style={{ fontSize: 11, color: "#64748b", letterSpacing: 1.5, textTransform: "uppercase", fontWeight: 700, marginBottom: 16 }}>
                  Fit Results
                </div>
                {fitResult ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    <div style={styles.metricCard}>
                      <div style={styles.metricLabel}>Peak Center</div>
                      <div style={styles.metricValue}>{fitResult.center.toFixed(4)}°</div>
                    </div>
                    <div style={styles.metricCard}>
                      <div style={styles.metricLabel}>FWHM</div>
                      <div style={{ ...styles.metricValue, color: scanMode === "omega" ? (fitResult.fwhm < 1 ? "#34d399" : fitResult.fwhm < 2 ? "#fbbf24" : "#f87171") : "#f1f5f9" }}>
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
                      <div style={{ ...styles.metricValue, color: fitResult.rSquared > 0.99 ? "#34d399" : fitResult.rSquared > 0.95 ? "#fbbf24" : "#f87171" }}>
                        {fitResult.rSquared.toFixed(6)}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div style={{ color: "#475569", fontSize: 13, textAlign: "center", padding: 20 }}>
                    Adjust fit range to see results
                  </div>
                )}
              </div>

              {/* Quality Badge (Omega mode only) */}
              {scanMode === "omega" && fitResult && (
                <div style={{ ...styles.card }}>
                  <div style={{ fontSize: 11, color: "#64748b", letterSpacing: 1.5, textTransform: "uppercase", fontWeight: 700, marginBottom: 12 }}>
                    Crystal Quality
                  </div>
                  <QualityBadge fwhm={fitResult.fwhm} />
                  <div style={{ fontSize: 11, color: "#64748b", marginTop: 12, lineHeight: 1.6 }}>
                    Target: FWHM &lt; 2° for low-angle grain boundary requirement
                  </div>
                </div>
              )}

              {/* Fit Parameters */}
              {fitResult && (
                <div style={{ ...styles.card }}>
                  <div style={{ fontSize: 11, color: "#64748b", letterSpacing: 1.5, textTransform: "uppercase", fontWeight: 700, marginBottom: 10 }}>
                    Fit Parameters
                  </div>
                  <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 2 }}>
                    <div>η (mixing) = <span style={{ color: "#e2e8f0" }}>{fitResult.eta?.toFixed(4)}</span></div>
                    <div>Background = <span style={{ color: "#e2e8f0" }}>{fitResult.bgA?.toFixed(2)} + {fitResult.bgB?.toFixed(4)}·x</span></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Empty State */}
        {!rawData && !error && (
          <div style={{ ...styles.card, textAlign: "center", padding: "60px 20px", marginBottom: 20 }}>
            <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.3 }}>⬡</div>
            <div style={{ fontSize: 14, color: "#64748b", maxWidth: 400, margin: "0 auto", lineHeight: 1.8 }}>
              Upload an XRD data file to begin analysis, or click <strong style={{ color: "#38bdf8" }}>Load Demo Data</strong> to explore with synthetic AlN (0002) data.
            </div>
          </div>
        )}

        {/* Sample Log Table */}
        {sampleLog.length > 0 && (
          <div style={{ ...styles.card }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <div style={{ fontSize: 11, color: "#64748b", letterSpacing: 1.5, textTransform: "uppercase", fontWeight: 700 }}>
                Sample Log ({sampleLog.length})
              </div>
              <button style={styles.btn("primary")} onClick={exportCSV}>↓ Export CSV</button>
            </div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr>
                    {["Sample", "Scan", "Center (°)", "FWHM (°)", "Intensity", "Quality", "Notes"].map(h => (
                      <th key={h} style={{ textAlign: "left", padding: "8px 12px", borderBottom: "1px solid rgba(100,116,139,0.2)", color: "#64748b", fontSize: 10, letterSpacing: 1.2, textTransform: "uppercase", fontWeight: 700 }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sampleLog.map((row, idx) => (
                    <tr key={row.id} style={{ background: idx % 2 === 0 ? "transparent" : "rgba(15,23,42,0.3)" }}>
                      <td style={{ padding: "8px 12px", color: "#e2e8f0" }}>{row.name}</td>
                      <td style={{ padding: "8px 12px", color: "#94a3b8" }}>{row.scanType}</td>
                      <td style={{ padding: "8px 12px", color: "#e2e8f0", fontVariantNumeric: "tabular-nums" }}>{row.center}</td>
                      <td style={{ padding: "8px 12px", color: "#e2e8f0", fontVariantNumeric: "tabular-nums" }}>{row.fwhm}</td>
                      <td style={{ padding: "8px 12px", color: "#e2e8f0", fontVariantNumeric: "tabular-nums" }}>{row.intensity}</td>
                      <td style={{ padding: "8px 12px" }}>
                        <span style={{
                          color: row.quality === "Excellent" ? "#34d399" : row.quality === "Acceptable" ? "#fbbf24" : row.quality === "Poor" ? "#f87171" : "#64748b",
                          fontWeight: 700
                        }}>{row.quality}</span>
                      </td>
                      <td style={{ padding: "8px 12px" }}>
                        <input type="text" value={row.notes} placeholder="Add notes…"
                          style={{ ...styles.input, width: "100%", background: "transparent", border: "1px solid rgba(100,116,139,0.15)" }}
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
        <div style={{ textAlign: "center", padding: "24px 0 12px" }}>
          <div style={{ fontSize: 10, color: "#334155", letterSpacing: 1.5, marginBottom: 8 }}>
            XRD PEAK ANALYZER · PSEUDO-VOIGT FITTING · AlN THIN FILM CHARACTERIZATION
          </div>
          <div style={{ fontSize: 10, color: "#475569", letterSpacing: 0.5 }}>
            © 2026 Zhaosen Qu
          </div>
        </div>
      </div>
    </div>
  );
}