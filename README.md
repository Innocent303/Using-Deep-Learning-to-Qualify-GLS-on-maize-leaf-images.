## The project in one paragraph
Maize leaf photos were annotated for grey leaf spot (GLS) lesions. Three models learn to outline leaf and lesion pixels.
Severity = lesion area / (lesion + leaf area). Models are compared with the Dice score (1 = perfect overlap),
first on a random 70/15/15 split, then by leaving one field site out at a time (LOGO), plus a small experiment on pretraining.

## Run order (my reading of the script headers, not tested end to end)
1. `01_data_prep`: inspect masks, merge and re-split the data 70/15/15, build site manifests and the leave-one-site-out folds, check all three datasets share the same split.
2. `02_train_standard_split`: U-Net (phase 2 sanity check, phase 3 training), Mask R-CNN and YOLO training.
3. `03_evaluate`: validation and test evaluation, side-by-side comparison and significance tests, inference-speed benchmark.
4. `04_leave_one_site_out`: train and evaluate 3 models x 4 held-out sites, then collect the results into one table.
5. `05_pretraining_experiment`: build the control image set, pretrain the encoders, run the four pretraining arms, analyse convergence.
6. `06_collect_for_stats`: gather per-seed results into CSV files for the statistical tests


