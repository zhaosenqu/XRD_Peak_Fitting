# XRD Peak Analyzer — Thin-Film X-ray Characterization

Browser-based analysis of X-ray diffraction and reflectivity data, with every metric computed from scratch in pure JS (no scientific libraries). Three modes — **θ–2θ**, **ω rocking curve**, and **XRR reflectivity** — switchable from the header tabs, in a clean light/academic interface.
Access here: https://zhaosenqu.github.io/XRD_Peak_Fitting/

## Scan Modes

### θ–2θ & ω rocking curve — Pseudo-Voigt peak fitting

Levenberg–Marquardt optimization of a Pseudo-Voigt profile (Gaussian + Lorentzian mix with a free η mixing parameter and a linear background), ~800 iterations, reporting peak center, FWHM, peak intensity, integrated area, and R² — plus the fitted η and background.

- **θ–2θ:** optional log-scale Y axis (e.g. for AlN 0002 near 36°).
- **ω rocking curve:** color-coded crystal-quality badge — Excellent (FWHM < 1°), Acceptable (< 2°), Poor (≥ 2°) — with the low-angle grain-boundary threshold note.

### XRR reflectivity — density & thickness

Automatic analysis of a specular reflectivity curve:

- **Critical angle → density.** θc is located at the half-plateau crossing of the total-reflection edge, then converted to electron density via `θc² = (rₑ·λ²/π)·nₑ` and to mass density via `ρ = nₑ / (N_A · Z/A)`. Cross-checked against silicon (≈ 2.33 g/cm³).
- **Thickness → Kiessig fringes.** Fringe maxima are detected on a lightly smoothed curve; thickness follows from `t = λ / (2·median(Δq))` with `q = √(sin²θ − sin²θc)` (refraction-corrected fringe spacing).
- **Inputs:** X-ray wavelength λ (default Cu Kα, 1.5406 Å) and the Z/A ratio (default 0.5). Angles are interpreted as incidence angle θ.

## Interactive Plot

Recharts ComposedChart:

- **Peak modes:** raw data as scatter points, the smooth Pseudo-Voigt fit in blue, a dashed amber line at the peak center, and a shaded fit-range region adjustable via two number inputs.
- **XRR mode:** log-scale reflectivity curve with a dashed line at the detected critical angle and amber markers on each Kiessig fringe.

## File Parsing

Two-column files separated by whitespace, commas, or semicolons; `#` and `!` comment lines are skipped, with friendly errors on failure. Drag-and-drop or click-to-browse upload.

## Demo Data

Synthetic data with counting (Poisson-like) noise:

- **ω rocking curve:** 500 pts, FWHM 1.8° centered at 0°.
- **θ–2θ:** 500 pts, peak at 36.04°, FWHM 0.35°.
- **XRR:** 800 pts, Fresnel critical edge + Kiessig oscillations (θc ≈ 0.30°, thickness 480 Å).

## Sample Log

In the peak-fitting modes, each fit can be appended to a session table with editable notes and exported as CSV.

## Run Locally

```bash
npm install
npm run dev      # Vite dev server with HMR
npm run build    # production build → dist/
npm run deploy   # build, then publish dist/ to the gh-pages branch
```
