# Freqdiag

`freqdiag.py` is a command-line diagnostic tool for analysing Fourier frequency lists and comparing their observed patterns with those expected from **linear beating, modulation, and nonlinear coupling**.

The program is intended primarily for the analysis of pulsating stars with close secondary frequencies or multiplet structures, such as RR Lyrae and high-amplitude Delta Scuti (HADS) stars.

**Freqdiag is a diagnostic tool, not a physical classifier.** It tests frequency, amplitude, and phase relations required by several mathematical descriptions. Compatibility with a model does not by itself prove the corresponding physical interpretation.

## Requirements

- Python 3.10 or later
- NumPy
- pandas

The Python dependencies can be installed with

```bash
pip install numpy pandas
```

or, if the repository contains `requirements.txt`,

```bash
pip install -r requirements.txt
```

No installation of `freqdiag` itself is required.

## Basic usage

```bash
python freqdiag.py frequencies.csv --baseline 200 --outdir diag_out
```

or, if the file is executable,

```bash
./freqdiag.py frequencies.csv --baseline 200 --outdir diag_out
```

The observing time span supplied with `--baseline` is in days. If no explicit frequency tolerance is given with `--tol`, the matching tolerance is calculated as

```text
tol = tol_factor / baseline
```

with `tol_factor = 1` by default.

For all command-line options:

```bash
python freqdiag.py --help
```

The program version can be displayed with

```bash
python freqdiag.py --version
```

## Input format

The input is a frequency table containing at least

- frequency,
- amplitude,
- phase.

By default, the corresponding column names are

```text
frequency amplitude phase
```

For example, a comma-separated input file may look like

```text
frequency,amplitude,phase
2.053612,0.2411,0.154
2.049069,0.0834,2.014
2.058155,0.0472,-0.735
4.107224,0.0913,-1.227
```

Frequencies are interpreted in `d^-1`. Amplitudes may be given in any consistent unit.

The input may be CSV, tab-separated, whitespace-separated, or a simple VizieR-style ASCII table. The separator is normally autodetected, but it can be specified explicitly, for example

```bash
--separator comma
--separator tab
--separator whitespace
```

### Different column names

Column names can be specified explicitly:

```bash
python freqdiag.py frequencies.dat \
    --separator whitespace \
    --freq-col Freq \
    --amp-col Amp \
    --phase-col Phi \
    --phase-unit deg \
    --baseline 150 \
    --outdir diag_out
```

Optional S/N and label columns can be specified with

```text
--snr-col
--label-col
```

### Headerless tables

Headerless tables are supported with

```bash
--no-header
```

The frequency, amplitude, and phase columns can then be selected using 1-based column numbers. For example,

```bash
python freqdiag.py frequencies.dat \
    --separator whitespace \
    --no-header \
    --freq-col 2 \
    --amp-col 4 \
    --phase-col 6 \
    --phase-unit cycles \
    --baseline 200
```

The equivalent explicit options

```text
--freq-col-index
--amp-col-index
--phase-col-index
```

are also available.

Initial lines of an input file can be skipped with

```text
--skiprows N
```

## Phase convention

Input phases can be supplied in radians, cycles, or degrees:

```bash
--phase-unit rad
--phase-unit cycles
--phase-unit deg
```

The default is radians.

The phase convention can be specified with

```bash
--phase-convention sine
--phase-convention cosine
--phase-convention unknown
```

The default is a sine-series convention. Cosine-series phases are converted internally to the equivalent sine-series phases.

For meaningful phase-relation diagnostics, all input phases must refer to the same epoch and use a consistent convention.

### Frequency-only analysis

If reliable phases are not available, phase-dependent diagnostics can be disabled with

```bash
--ignore-phases
```

or equivalently

```bash
--frequency-only
```

Frequency-grid tests and real-amplitude diagnostics are still performed, but phase and complex-amplitude diagnostics are disabled.

In this mode only the frequency and amplitude columns are required.

## Primary frequency

The primary pulsation frequency **f₀** can be supplied explicitly:

```bash
--f0 2.053612
```

If `--f0` is omitted, `freqdiag` selects the primary frequency automatically.

The default automatic method is

```text
--f0-auto-method harmonic-series
```

which ranks possible primary frequencies using the observed sequence

```text
f, 2f, 3f, ...
```

rather than simply assuming that the largest-amplitude peak is the primary frequency.

The legacy largest-amplitude selection can be requested with

```bash
--f0-auto-method strongest
```

The automatic harmonic-series ranking is written to

```text
freqdiag_f0_candidates.csv
```

when applicable.

For uncertain or unusual frequency spectra, supplying `--f0` explicitly is recommended.

## Candidate close frequencies

By default, the program searches automatically for close-frequency offsets associated with the primary harmonic sequence.

Candidate offsets may also be supplied manually with `--fb`. For example,

```bash
python freqdiag.py frequencies.csv \
    --f0 2.053612 \
    --fb 0.004543,-0.006210 \
    --baseline 200
```

Automatic candidate detection can be disabled with

```bash
--no-auto-fb
```

in which case only manually supplied candidates are used.

## What Freqdiag tests

The program examines several kinds of Fourier signatures.

### Linear beating

It tests whether a close secondary frequency **f′** is compatible with

- a sinusoidal secondary oscillation, or
- a non-sinusoidal secondary oscillation with harmonics **l f′**.

### Periodic modulation

It searches for equidistant multiplet structures of the form

```text
n f₀ ± L fₘ
```

and evaluates modulation-related amplitude and phase diagnostics, including sideband behaviour and higher side orders.

### Quadratic coupling

It tests quadratic combination-frequency relations produced by coupling between the primary pulsation and a close secondary oscillation.

Right- and left-side secondary hypotheses are tested separately. The diagnostics include frequency relations, amplitude ratios, combination phases, and complex-amplitude consistency where phase information is available.

If several secondary candidates are present, the program can also test quadratic secondary-secondary combination terms.

### Modulation versus coupling

An equidistant multiplet frequency grid alone does **not** distinguish modulation from quadratic coupling, because both descriptions can generate the same frequency positions.

Where sufficient amplitude and phase information is available, `freqdiag` therefore compares modulation and coupling using common complex side-peak data and additional side-order diagnostics.

The resulting classifications and scores are measures of **diagnostic compatibility, not probabilities**.

## Frequency matching

A frequency is considered matched if it lies within the adopted tolerance of an expected frequency.

The tolerance can be supplied directly:

```bash
--tol 0.0001
```

or derived from the observational baseline:

```bash
--baseline 200
```

By default,

```text
tol = 1 / baseline
```

Unassigned significant frequencies are retained in the output. They are **not automatically treated as evidence against a model**.

## Output

Unless another directory is specified with `--outdir`, results are written to

```text
freqdiag_output/
```

The main human-readable result is

```text
freqdiag_report.md
```

The complete output includes:

- `freqdiag_report.md` — human-readable diagnostic report
- `freqdiag_model_scores.csv` — full model and submodel score table
- `freqdiag_report_scores.csv` — model rows selected for the Markdown report
- `freqdiag_feature_flags.csv` — diagnostic feature and qualifier rows
- `freqdiag_matches.csv` — expected frequencies and their observed matches
- `freqdiag_frequency_assignments.csv` — input frequencies with assignment tags
- `freqdiag_harmonics.csv` — detected primary harmonics
- `freqdiag_f0_candidates.csv` — automatic primary-frequency ranking, when applicable
- `freqdiag_candidates.csv` — detected close-frequency offsets
- `freqdiag_phase_tests.csv` — quadratic-coupling amplitude and phase diagnostics
- `freqdiag_modulation_summary.csv` — modulation and AM/FM/PM subdiagnostics
- `freqdiag_side_order_sequences.csv` — signed higher-side-order amplitude sequences
- `freqdiag_coupling_summary.csv` — coupling diagnostics
- `freqdiag.json` — machine-readable summary
- `freqdiag_latex_table.tex` — compact LaTeX summary table

The Markdown report is normally the best starting point for inspecting a run, while the CSV files retain the detailed diagnostics.

## Example: VizieR/ASCII input

A typical analysis of a whitespace-separated table with non-standard column names is

```bash
python freqdiag.py V1127_Aql.txt \
    --separator whitespace \
    --freq-col Freq \
    --amp-col Amp \
    --phase-col Phi \
    --phase-unit deg \
    --baseline 150 \
    --nmax 20 \
    --outdir diag_V1127
```

## Citation

If you use `freqdiag` in scientific work, please cite the paper describing the diagnostic method.

Benkő J. M. and Plachy E.: Beating and coupling in pulsating stars: a unified Fourier description and observational diagnostics, Astronomy and Astrophysics (submitted)
Full citation information will be added after acceptance/publication of the accompanying paper.

## License

`freqdiag` is distributed under the MIT License. See the `LICENSE` file for details.

## Author

**József M. Benkő**
