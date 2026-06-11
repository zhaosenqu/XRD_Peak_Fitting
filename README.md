# XRD Peak Analyzer — Thin-Film X-ray Characterization

Browser-based analysis of X-ray diffraction and reflectivity data, with every metric computed from scratch in pure JS (no scientific libraries). Three modes — **θ–2θ**, **ω rocking curve**, and **XRR reflectivity** — switchable from the header tabs.

Access here: https://zhaosenqu.github.io/XRD_Peak_Fitting/

## Input File Format

Upload a plain-text two-column file: **angle** (degrees) in the first column, **intensity** (counts) in the second. At least 5 data rows are required.

**Accepted separators:** whitespace (spaces or tabs), commas, or semicolons.  
**Comment lines** starting with `#` or `!` are skipped.  
**Accepted extensions:** `.xy`, `.csv`, `.txt`, `.dat`

### θ–2θ example

```
# AlN 0002 reflection, Cu Kα
# 2theta   counts
34.000      142
34.016      155
34.032      174
...
36.040    24831
...
38.000      138
```

### ω rocking curve example

```
# AlN 0002 rocking curve
# omega    counts
-5.000       62
-4.980       71
-4.960       84
...
 0.000    11943
...
 5.000       58
```

The peak should be centered near ω = 0°. Use the **Fit Range** inputs to exclude tails if the background is noisy.

### XRR example

```
# Specular reflectivity, Cu Kα (λ = 1.5406 Å)
# theta    counts
0.050    982541
0.054    975302
0.058    968844
...
0.300    412000   # near critical angle
...
3.000        18
```

Angles are **incidence angle θ** (not 2θ). The scan must start *below* the critical angle (typically < 0.1°) so the total-reflection plateau is visible — the analysis locates θc from the half-plateau crossing. Set the **Z/A ratio** to the electron-to-nucleon ratio of your film material (e.g. Al: 13/26.98 ≈ 0.482; AlN: ~0.492; Si: 14/28.09 ≈ 0.498; default 0.5 is a reasonable estimate for most thin-film materials).

## Scan Modes

### θ–2θ & ω rocking curve — Pseudo-Voigt peak fitting

Levenberg–Marquardt optimization of a Pseudo-Voigt profile (Gaussian + Lorentzian mix with a free η mixing parameter and a linear background), ~800 iterations, reporting peak center, FWHM, peak intensity, integrated area, and R² — plus the fitted η and background coefficients.

- **θ–2θ:** optional log-scale Y axis.
- **ω rocking curve:** color-coded crystal-quality badge — Excellent (FWHM < 1°), Acceptable (< 2°), Poor (≥ 2°) — based on the low-angle grain-boundary threshold.

### XRR reflectivity — density & thickness

Automatic extraction from a specular reflectivity curve:

- **Critical angle → density.** θc is located at the half-plateau crossing, then converted to electron density via `θc² = (rₑ·λ²/π)·nₑ` and to mass density via `ρ = nₑ / (N_A · Z/A)`.
- **Thickness → Kiessig fringes.** Fringe maxima are detected on a smoothed curve; thickness follows from `t = λ / (2·median(Δq))` with `q = √(sin²θ − sin²θc)` (refraction-corrected).
- **Inputs:** X-ray wavelength λ (default Cu Kα, 1.5406 Å) and Z/A ratio (default 0.5).

## Sample Log

In the peak-fitting modes, fits can be appended to a session table with editable notes, then exported as CSV or cleared.

## Run Locally

```bash
npm install
npm run dev      # Vite dev server with HMR
npm run build    # production build → dist/
```
