---
name: summarize-fact-gs-results
description: Aggregate comparable FaCT-GS reconstruction, warm-start, compression, scaling, initialization, or Gaussian-count experiment records into publication-ready LaTeX tables. Use for benchmark summaries, cross-method comparisons, best-value highlighting, paper-style tables, or LaTeX/CSV output from collect-fact-gs-results and experiments helper outputs.
---

# Summarize FaCT-GS Results

1. Read structured records from `collect-fact-gs-results` or the matching `experiments/helpers` collector. Reconcile task, dataset, split, metric definition, stopping condition, and units before comparing runs.
2. Choose columns appropriate to the study. Reconstruction defaults to `METHOD`, `PSNR$_{3D}$`, `SSIM$_{3D}$`, `PROJ_PSNR`, `PROJ_SSIM`, and `TIME(min)`. Compression may add Gaussian count/model size; scaling may emphasize resolution and time.
3. Keep missing values as `/`. Select maxima for PSNR/SSIM and minima for time/size using unrounded values. Preserve ties and never highlight incomparable or missing entries.
4. Start from `assets/results-table.tex`, escape labels, state rounding, and include `booktabs` plus `\newcommand{\best}[1]{\textbf{#1}}` when absent.
5. Save the `.tex` and source CSV/JSON together. Check brace balance and compile a non-destructive preview when a LaTeX engine is available.
