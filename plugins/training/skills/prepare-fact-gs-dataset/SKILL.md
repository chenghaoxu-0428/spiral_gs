---
name: prepare-fact-gs-dataset
description: Prepare, validate, visualize, and troubleshoot FaCT-GS CT datasets in the R2-Gaussian meta_data.json layout. Use for synthetic or real cone/parallel-beam data, raw DICOM volume/projection conversion, spiral or stitch acquisition generation, the repository's data_preprocess/norm_pipeline.py workflow, initialization arrays, or checking whether data is usable by train_recon.py or train_volume.py.
---

# Prepare FaCT-GS Dataset

## Choose the supported path

1. Inspect the repository and existing data before asking questions. FaCT-GS consumes the R2-Gaussian layout but integrates reconstruction initialization into `train_recon.py`.
2. Prefer `data_preprocess/norm_pipeline.py` when it exists. Read `data_preprocess/README_norm.md` and the example YAML before editing a run config. Otherwise follow the appropriate generator README under `fact_gs/r2_gaussian/data_generator/{synthetic_dataset,real_dataset}/`.
3. Resolve the exact Python interpreter, dataset type (`syn` or `real`), raw inputs, scanner geometry, acquisition (`spiral`, `stitch`, or both), split sizes, seed, and target model label. Never invent scanner geometry.
4. For the normalized pipeline, keep run YAMLs under `data_preprocess/configs/` and use:
   `data/{real|syn}/{organ}/{spiral|stitch}/ntrain<N>/<model>/`
   Preserve an existing legacy/project dataset path rather than migrating it without a request.

## Validate before generation

1. For an existing reconstruction dataset, require readable `meta_data.json`, non-empty training projections, and scanner/camera metadata consistent with that file. Check test projections when `model.eval=true`. Treat `init_*.npy` as optional for `model.init_mode=auto`.
2. For volume fitting, additionally require `<model.vol_name>.npy` or `<model.vol_name>.tiff` (`vol_prior` by default, `vol_gt` for compression).
3. When ground truth is expected, verify `vol_gt.npy` or `vol_gt.tiff`; do not require it for real scans without a trustworthy reference.
4. With the normalized pipeline, run the read-only check first:
   ```bash
   <python> data_preprocess/norm_pipeline.py --config data_preprocess/configs/<run>.yml --validate-only
   ```
5. Show validation findings and the exact generation command, then obtain confirmation before DICOM loading, projection generation, FDK, or overwriting output.
6. Run the same command without `--validate-only`, verify the resolved output directory, parse `meta_data.json`, inspect array shapes/finiteness, and record the exact config.

## Initialization and visualization

- Prefer `model.init_mode=auto` for reconstruction. A generated `init_<dataset-name>.npy` is reusable but not mandatory; it must contain finite `[x,y,z,density]` rows.
- Use `gradient` or `intensity` for explicit online FDK initialization. Use `precomputed` only when the expected init file exists. Use `prior` only after a successful volume-fitting run.
- Offer `spiral_tools/scripts/visualize_scene.py -s <dataset>` when available. It is interactive: request GUI permission, wait for exit, and ask whether geometry and projections look correct.
- If visualization is rejected, diagnose orientation, handedness, angle/z ordering, detector axes, and unit conversion before regenerating.
