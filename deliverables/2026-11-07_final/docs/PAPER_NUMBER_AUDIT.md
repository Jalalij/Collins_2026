# Number audit of the TWC paper

Every macro below is written by `modeling/experiments/fr_paper_numbers.py` into `paper_twc/figs/data/numbers_story.tex`. The source column names the result file the value is read from. The older macros (`numbers_audit.tex`, `numbers_results.tex`, `numbers_analysis.tex`) are written by `final_audit_numbers.py` and `final_tables.py` from `modeling/results_final/summary/`.

| Macro | Value | Source |
|---|---|---|
| `\DataSessions` | 996 | results_final/analysis/dataset_stats.json |
| `\DataRows` | 2.90 | results_final/analysis/dataset_stats.json (million rows) |
| `\DataHours` | 161 | results_final/analysis/dataset_stats.json |
| `\DataDaysOW` | 10 | results_final/analysis/dataset_stats.json |
| `\DataDaysSL` | 6 | results_final/analysis/dataset_stats.json |
| `\DataDaysQ` | 13 | results_final/analysis/dataset_stats.json |
| `\DataHoursOWDL` | 9.5 | results_final/analysis/dataset_stats.json |
| `\DataHoursOWUL` | 9.4 | results_final/analysis/dataset_stats.json |
| `\DataHoursSLDL` | 8.6 | results_final/analysis/dataset_stats.json |
| `\DataHoursSLUL` | 8.6 | results_final/analysis/dataset_stats.json |
| `\DataHoursQDL` | 62.8 | results_final/analysis/dataset_stats.json |
| `\DataHoursQUL` | 62.3 | results_final/analysis/dataset_stats.json |
| `\MaxRMSEGain` | 5 | results_final/summary/summary.csv (largest RMSE reduction vs S2S-E, any model, any link) |
| `\PersRatioLo` | 1.3 | results_final/summary/summary.csv |
| `\PersRatioHi` | 2.4 | results_final/summary/summary.csv |
| `\IsProtoMaxAbs` | 2.3 | results_final/summary/summary.csv rtt_MIS_y2style (test-residual intervals) |
| `\VolDmaxAbs` | 0.12 | results_final/analysis/volatility/acf_summary.csv d_max_abs_lag0p4to10s |
| `\VolAbsdLo` | 0.08 | results_final/analysis/volatility/acf_summary.csv absd_1s_measured_within |
| `\VolAbsdHi` | 0.10 | results_final/analysis/volatility/acf_summary.csv |
| `\VolSLDLlevel` | 0.84 | results_final/analysis/volatility/acf_summary.csv rtt_1s_measured_within |
| `\VolSLDLlevelMin` | 0.76 | results_final/analysis/volatility/acf_summary.csv rtt_1s_sessions_min |
| `\VolSLDLnSess` | 37 | results_final/analysis/volatility/acf_summary.csv |
| `\VolSignRevLo` | 64 | results_final/analysis/volatility/window_corr.csv |
| `\VolSignRevHi` | 74 | results_final/analysis/volatility/window_corr.csv |
| `\VolSignNNLo` | 63 | results_final/analysis/volatility/window_corr.csv |
| `\VolSignNNHi` | 78 | results_final/analysis/volatility/window_corr.csv |
| `\VolWithinLo` | 0.00 | results_final/analysis/volatility/window_corr.csv corr_inputsd_abs_err_s2s_e_within |
| `\VolWithinHi` | 0.03 | results_final/analysis/volatility/window_corr.csv |
| `\VolBetweenLo` | 0.48 | results_final/analysis/volatility/window_corr.csv corr_inputsd_abs_err_s2s_e_between |
| `\VolBetweenHi` | 0.89 | results_final/analysis/volatility/window_corr.csv |
| `\VolSLDLrv` | 0.73 | results_final/analysis/volatility/window_corr.csv corr_input_rv_horizon_rv_within |
| `\SkewQUL` | 3.5 | results_final/main/quectel_ul/preds (S2S-E calibration-session error at 1 s, q.975 / /q.025/) |
| `\SkewOWDL` | 3.0 | results_final/main/oneweb_dl/preds (S2S-E calibration-session error at 1 s, q.975 / /q.025/) |
| `\CovConstCalmLo` | 0.948 | results_final/analysis/condcov/summary.csv |
| `\CovConstCalmHi` | 0.996 | results_final/analysis/condcov/summary.csv |
| `\CovConstVolLo` | 0.856 | results_final/analysis/condcov/summary.csv |
| `\CovConstVolHi` | 0.947 | results_final/analysis/condcov/summary.csv |
| `\CovConstSLDLdOne` | 0.996 | results_final/analysis/condcov/summary.csv |
| `\CovConstSLDLdTen` | 0.947 | results_final/analysis/condcov/summary.csv |
| `\CovConstQULdOne` | 0.963 | results_final/analysis/condcov/summary.csv |
| `\CovConstQULdTen` | 0.936 | results_final/analysis/condcov/summary.csv |
| `\CovNormOWDLdOne` | 0.772 | results_final/analysis/condcov/summary.csv |
| `\GapDSSLDL` | 0.026 | results_final/analysis/condcov/summary.csv maxdev |
| `\GapConstSLDL` | 0.048 | results_final/analysis/condcov/summary.csv |
| `\GapDSQDL` | 0.006 | results_final/analysis/condcov/summary.csv maxdev |
| `\GapConstQDL` | 0.027 | results_final/analysis/condcov/summary.csv |
| `\GapDSQUL` | 0.003 | results_final/analysis/condcov/summary.csv maxdev |
| `\GapConstQUL` | 0.014 | results_final/analysis/condcov/summary.csv |
| `\WidRatioOWDL` | 1.30 | results_final/analysis/condcov/summary.csv wid_ratio_d10_d1 |
| `\CovDSOWDLdTen` | 0.888 | results_final/analysis/condcov/summary.csv |
| `\CovNormTunedOWDLdTen` | 0.952 | results_final/analysis/condcov/summary.csv |
| `\ErrRatioOWDL` | 4.3 | reports/final_report/data/fr_condcov_wid_oneweb_dl.dat (per-decile calibrated width at 1 s, d10/d1) |
| `\WidRatioOWUL` | 1.04 | results_final/analysis/condcov/summary.csv wid_ratio_d10_d1 |
| `\CovDSOWULdTen` | 0.899 | results_final/analysis/condcov/summary.csv |
| `\CovNormTunedOWULdTen` | 0.951 | results_final/analysis/condcov/summary.csv |
| `\ErrRatioOWUL` | 2.0 | reports/final_report/data/fr_condcov_wid_oneweb_ul.dat (per-decile calibrated width at 1 s, d10/d1) |
| `\WidConstSLDL` | 163 | reports/final_report/data/fr_condcov_wid_starlink_dl.dat |
| `\WidDSSLDLcalmLo` | 16 | fr_condcov_wid_starlink_dl.dat deciles 1-3 |
| `\WidDSSLDLcalmHi` | 19 | fr_condcov_wid_starlink_dl.dat deciles 1-3 |
| `\WidDSSLDLloadLo` | 119 | fr_condcov_wid_starlink_dl.dat deciles 4-10 |
| `\WidDSSLDLloadHi` | 134 | fr_condcov_wid_starlink_dl.dat deciles 4-10 |
| `\DecCentreLo` | -2.2 | results_final/analysis/condcov/summary.csv ds2s_meanconst vs s2s_const |
| `\DecCentreHi` | 0.9 | results_final/analysis/condcov/summary.csv |
| `\DecWidthLoAbs` | 1.4 | results_final/analysis/condcov/summary.csv s2s_ds2swidth vs s2s_const |
| `\DecWidthHiAbs` | 29.7 | results_final/analysis/condcov/summary.csv |
| `\DecAsymLoAbs` | 4.3 | results_final/analysis/condcov/summary.csv ds2s_cqr vs s2s_ds2swidth |
| `\DecAsymHiAbs` | 27.3 | results_final/analysis/condcov/summary.csv |
| `\AsymQULratio` | 10 | results_final/analysis/condcov/summary.json info.ds2s_cqr.asym_h5 |
| `\ExtBestPointLoAbs` | 6 | results_final/analysis/condcov/summary.csv best point-forecast interval vs ds2s_cqr |
| `\ExtBestPointHiAbs` | 44 | results_final/analysis/condcov/summary.csv |
| `\ExtFrfullLo` | -3.1 | results_final/analysis/condcov/summary.csv frfull_const vs s2s_const |
| `\ExtFrfullHi` | 0.7 | results_final/analysis/condcov/summary.csv |
| `\ExtLQRMaxAbs` | 6.2 | results_final/analysis/condcov/summary.csv /lqr_cqr vs ds2s_cqr/ |
| `\GaussSentence` | Third, Gaussian heads do not recover the gain. GS2S-E, whose quantiles are symmetric, has a higher interval score than DS2S-E on 6 of the six links (+4.3\% to +32.5\%), with the largest gap on 5G UL. | results_final/analysis/condcov/summary.csv gauss_cqr vs ds2s_cqr |
| `\GaussSeedNote` | uses 3 seeds | results_final/analysis/condcov/summary.csv |
| `\BSISWorstAbs` | 9.4 | results_final/analysis/bootstrap/bootstrap_summary.csv |
| `\BSISBestAbs` | 41.1 | results_final/analysis/bootstrap/bootstrap_summary.csv |
| `\BSBestMAEAbs` | 17.8 | results_final/analysis/bootstrap/bootstrap_summary.csv |
| `\BSMAEsigLoAbs` | 1.6 | results_final/analysis/bootstrap/bootstrap_summary.csv |
| `\BSMAEsigHiAbs` | 17.8 | results_final/analysis/bootstrap/bootstrap_summary.csv |
| `\BSRMSEMaxAbs` | 5.0 | results_final/analysis/bootstrap/bootstrap_summary.csv |
| `\BSnSessTot` | 156 | results_final/analysis/bootstrap/bootstrap_summary.csv |
| `\BSWidRatioLo` | 2.2 | results_final/analysis/bootstrap/bootstrap_summary.csv width_ratio_session_vs_naive (main rows) |
| `\BSWidRatioHi` | 15.2 | results_final/analysis/bootstrap/bootstrap_summary.csv |
| `\BSLooMax` | 2.4 | results_final/analysis/bootstrap/bootstrap_summary.csv loo_min_pct / loo_max_pct (IS) |
| `\BSnWinTot` | 444{,}844 | results_final/main/*/preds/truth.npz |
| `\MeanMAELo` | -0.7 | preds npz (MAE of the DS2S-E mean vs S2S-E) |
| `\MeanMAEHi` | 3.1 | preds npz |
| `\MedMAELo` | -17.8 | preds npz (MAE of the DS2S-E median vs S2S-E) |
| `\MedMAEHi` | -1.6 | preds npz |
| `\MeanMAESLDL` | -11.6 | preds npz |
| `\LQRRMSELoAbs` | 0.2 | results_final/analysis/bootstrap/main_vs_others.csv (against lqr_fr) |
| `\LQRRMSEHiAbs` | 11.4 | results_final/analysis/bootstrap/main_vs_others.csv (against lqr_fr) |
| `\LQRsigLo` | 2 | results_final/analysis/bootstrap/main_vs_others.csv (against lqr_fr) |
| `\LQRsigHi` | 6 | results_final/analysis/bootstrap/main_vs_others.csv (against lqr_fr) |
| `\LQRsigLinks` | Starlink DL, 5G DL, 5G UL | results_final/analysis/bootstrap/main_vs_others.csv (against lqr_fr) |
| `\LeakOptDSMAEhOne` | 20 | results_final/analysis/leak/leak_summary.json |
| `\LeakOptDSMAEavg` | 6.8 | results_final/analysis/leak/leak_summary.json |
| `\LeakOptDSRMSEhOne` | 11 | results_final/analysis/leak/leak_summary.json |
| `\LeakOptSSMAEhOne` | 5.4 | results_final/analysis/leak/leak_summary.json |
| `\LeakDSAhOne` | 14.6 | results_final/analysis/leak/leak_summary.json |
| `\LeakDSBhOne` | 17.5 | results_final/analysis/leak/leak_summary.json |
| `\LeakDSChOne` | 15.0 | results_final/analysis/leak/leak_summary.json |
| `\TraceWidS` | 163 | results_final/analysis/trace/trace_starlink_dl.json |
| `\TraceWidDCalm` | 14 | results_final/analysis/trace/trace_starlink_dl.json |
| `\TraceWidDLoad` | 144 | results_final/analysis/trace/trace_starlink_dl.json |
| `\SchedDSLDL` | 198 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedSSdupSLDL` | 33.2 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedSSmissSLDL` | 1.74 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedDSdupSLDL` | 15.1 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedDSmissSLDL` | 1.53 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedSaveSLDL` | 55 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedSaveSLDLlo` | 49 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedSaveSLDLhi` | 59 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedSaveSLUL` | -1.6 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedSaveOWUL` | +0.4 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedDSdupOWDL` | 50.2 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedSSdupOWDL` | 51.3 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedDSdupCurveOWDL` | 56.7 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedDSmissOWDL` | 1.03 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedSSmissOWDL` | 0.83 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedFFdupSLDL` | 24.8 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedDSdupCurveSLDL` | 13.4 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedSSdupCurveSLDL` | 34.2 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\SchedSaveFFSLDL` | 46 | results_final/analysis/scheduler/headline.csv (D90, one95) |
| `\GaussSchedSentence` | GS2S-E, whose quantiles are symmetric, needs 25.6\% at the same miss rate, close to the point forecaster S2S-AE-E, so a symmetric spread adds little. | results_final/analysis/scheduler/link_starlink_dl.json decomposition_at_s2s95_miss.gauss |
| `\BackupDLmax` | 0.01 | results_final/analysis/scheduler/backup_check.csv |
| `\BackupQULSL` | 7.8 | results_final/analysis/scheduler/backup_check.csv |
| `\BackupSSmissSLUL` | 1.45 | results_final/analysis/scheduler/backup_check.csv |
| `\BackupSSmissCorrSLUL` | 1.70 | results_final/analysis/scheduler/backup_check.csv |
| `\BackupDSmissSLUL` | 1.56 | results_final/analysis/scheduler/backup_check.csv |
| `\BackupDSmissCorrSLUL` | 1.80 | results_final/analysis/scheduler/backup_check.csv |
| `\RiskMeanLo` | 91 | results_final/analysis/scheduler/riskmap_starlink_dl.csv (mean decile 5) |
| `\RiskMeanHi` | 93 | results_final/analysis/scheduler/riskmap_starlink_dl.csv (mean decile 5) |
| `\RiskLowPct` | 0.7 | results_final/analysis/scheduler/riskmap_starlink_dl.csv (mean decile 5) |
| `\RiskHighPct` | 12.3 | results_final/analysis/scheduler/riskmap_starlink_dl.csv (mean decile 5) |
| `\BoundCovLo` | 0.936 | results_final/analysis/scheduler/calibrated_<link>.csv coverage_of_U |
| `\BoundCovHi` | 0.960 | results_final/analysis/scheduler/calibrated_<link>.csv coverage_of_U |
| `\GradLastLo` | 42 | results_final/analysis/attention/attn_summary.csv grad_sum_change_mass_last1s |
| `\GradLastHi` | 66 | results_final/analysis/attention/attn_summary.csv |
| `\AttnAblMax` | 0.8 | results_final/analysis/attention/attn_summary.csv |
| `\AttnAblOWDL` | 12 | results_final/analysis/attention/attn_summary.csv |
| `\AttnAblOWULmap` | 17 | results_final/analysis/attention/attn_summary.csv |
| `\AttnSinkLo` | 45 | attn_summary.json profile_over_steps_x30[29]/30 |
| `\AttnSinkHi` | 63 | attn_summary.json |
| `\GradEngLo` | 34 | results_final/analysis/attention/attn_summary.csv grad_h15_eng_feature_share |
| `\GradEngHi` | 76 | results_final/analysis/attention/attn_summary.csv |
| `\SweepISLoAbs` | 7 | results_final/summary/sweeps.csv |
| `\SweepISHiAbs` | 41 | results_final/summary/sweeps.csv |
| `\SweepRMSELo` | -7.8 | results_final/summary/sweeps.csv |
| `\SweepRMSEHi` | 2.1 | results_final/summary/sweeps.csv |
| `\WeightPrefLinks` | OneWeb DL, OneWeb UL, Starlink DL, Starlink UL, 5G DL, 5G UL | preds npz (calibration-session pinball loss) |
| `\WeightSentence` | A pinball weight $\omega=4$, which the calibration sessions prefer on all six links, changes the MAE of DS2S by -10.7\% to -0.3\%, the interval score by -4.4\% to +1.1\%, and the RMSE by -0.7\% to +1.2\% (means of 3 seeds). | results_final/summary/runs.csv (fr_dist_w4 vs fr_dist) |
| `\JitISRatio` | 3.0 | reports/final_report/data/numbers_results.tex JitIS* |
