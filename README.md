# Fitting Engine — Levenberg-Marquardt Optimization

Levenberg-Marquardt optimization of the Pseudo-Voigt profile (Gaussian + Lorentzian mix with free η parameter and linear background), ~800 iterations, computing R², integrated area, and all peak metrics from scratch in pure JS.
Access Here: https://zhaosenqu.github.io/XRD_Peak_Fitting/

## Two Scan Modes

**θ–2θ** (with optional log-scale Y axis, auto-detecting AlN 0002 near 36°) and **ω rocking curve**, switchable via header tabs. The rocking curve mode includes the color-coded crystal quality badge (green/yellow/red) with the grain boundary threshold note.

## Interactive Plot

Recharts ComposedChart showing:
- Raw data as scatter dots
- The smooth fit curve in cyan
- A dashed reference line at peak center
- A shaded fit-range region
- Adjustable fit range via two number inputs

## File Parsing

Handles whitespace- or comma-separated two-column files, skips `#`/`!` comment lines, and shows friendly errors on failure. Drag-and-drop or click-to-browse upload.

## Demo Data

Generates 500-point synthetic AlN data with Poisson noise:
- **Rocking curve:** FWHM=1.8° centered at 0°
- **θ-2θ:** peak at 36.04° with FWHM=0.35°

## Sample Log

Each fit can be appended to a session table with editable notes, exportable as CSV via Blob download.
