# Reference notes: final report and IEEE TWC paper

- `refs_verified.bib` holds the **38 entries selected** for the TWC paper. The cap is 40, so 2 slots are spare.
- `refs_pool.bib` holds the **full verified pool of 65 entries**. This is the selected set plus alternates, and the final report can draw on all of it.

## How the entries were verified (2026-11-01)

- **Entries with a DOI:** the DOI was resolved through the Crossref REST API (`https://api.crossref.org/works/<doi>`), and the title, authors, venue, year and pages were checked against the BibTeX. All 54 DOIs in the pool resolved and matched.
- **Preprints and ML-venue entries:** these were checked against their `https://arxiv.org/abs/<id>` landing pages. That covers all 17 arXiv ids cited in the pool, including the 2026 preprints chi2026reading and casparsen2026statistical.
- **NeurIPS/ICML entries without a DOI:** these were checked against the NeurIPS proceedings or PMLR pages.
- **Unverified fields:** no DOI was invented. Volume and pages for minovski2023throughput could not be re-checked because Crossref holds only the early-access record. That entry is a pool alternate, not in the selected set.
- **ARA testbed key:** the initial keys `ISLAM2025111188` / `islam_design_2024` are now **`islam2025design`**. It is the published version: *Computer Networks* 263:111188 (2025), preprint arXiv:2408.00913.
- **Duplicate removed:** liu2025vivisecting appeared twice in the candidate list, under leo_measurement and link_prediction, and is kept once.

## Selected set (38)

| key | topic | one-line relevance | verify URL |
|---|---|---|---|
| islam2025design | testbed | ARA wireless living lab (Iowa State) where the 5 Hz Starlink/OneWeb/5G link-state data were collected. | https://doi.org/10.1016/j.comnet.2025.111188 |
| michel2022first | leo_measurement | Canonical first Starlink measurement (latency, throughput, loss under load); motivates LEO link-state variability. | https://doi.org/10.1145/3517745.3561416 |
| mohan2024multifaceted | leo_measurement | Documents globally synchronized 15 s Starlink reconfigurations driving RTT/throughput jumps; motivates periodic features. | https://doi.org/10.1145/3589334.3645328 |
| tanveer2023making | leo_measurement | Dish telemetry reveals Starlink's 15 s global scheduler slots, explaining handover-driven regime changes we forecast. | https://doi.org/10.1145/3624354.3630586 |
| pan2023measuring | leo_measurement | Long-running Starlink campaign (access, PoP, backbone) characterizing latency dynamics and periodic handovers. | https://doi.org/10.1109/PIMRC56721.2023.10294034 |
| zhao2025measuring | leo_measurement | Cross-layer OneWeb measurement; ISL-less handovers cause large latency swings, the OneWeb dynamics we predict. | https://doi.org/10.23919/TMA66427.2025.11096999 |
| perrin2025measurement | leo_measurement | Iowa State/ARA OneWeb dynamics and liquid data transport study; direct predecessor of our OneWeb data and multipath use case. | https://doi.org/10.1109/MILCOM64451.2025.11310330 |
| liu2025vivisecting | leo_measurement | Multi-month Starlink throughput measurement and learned prediction; closest prior work and key comparison point. | https://doi.org/10.1145/3768971 |
| garcia2023multitimescale | leo_measurement | Starlink throughput variability from sub-second/15 s to diurnal scales; justifies multi-scale rolling features and short horizons. | https://doi.org/10.1145/3614204.3616108 |
| kassem2022browser | leo_measurement | Reports bursty Starlink packet loss correlated with handovers; motivates loss as a forecast target. | https://doi.org/10.1145/3517745.3561457 |
| narayanan2020lumos5g | link_prediction | Canonical context-aware ML throughput prediction for commercial 5G; terrestrial counterpart to LEO forecasting. | https://doi.org/10.1145/3419394.3423629 |
| mei2019realtime | link_prediction | Seminal LSTM short-horizon mobile bandwidth predictor; motivates recurrent design and strong classical baselines. | https://doi.org/10.1007/978-3-030-15986-3_3 |
| raca2020leveraging | link_prediction | Classical ML vs. deep learning for cellular throughput prediction; supports engineered features plus a DL comparison. | https://doi.org/10.1109/MCOM.001.1900394 |
| tian2025attention | link_prediction | Attention-based deep RTT predictor for LEO networks; directly comparable to our attention GRU Seq2Seq. | https://doi.org/10.1109/ICC52391.2025.11160911 |
| basit2024predictability | link_prediction | Limits of fine-grained (sub-second) cellular throughput predictability; supports reporting skill relative to persistence. | https://doi.org/10.1109/MASS62177.2024.00018 |
| benghe2026horizon | link_prediction | Global, coarse-grained learned Starlink latency/throughput predictor; contrasts with our 5 Hz short-horizon forecasts. | https://doi.org/10.1145/3805639 |
| maglione2025intelligent | link_prediction | LSTM path-state forecasts feeding a multi-access scheduler; direct precedent for prediction-aware MPQUIC scheduling. | https://doi.org/10.1109/VTC2025-Spring65109.2025.11174823 |
| cho2014learning | forecasting_methods | Introduces the GRU used in our encoder and decoder. | https://doi.org/10.3115/v1/D14-1179 |
| sutskever2014sequence | forecasting_methods | Encoder-decoder (Seq2Seq) framework underlying our multi-step forecaster. | https://arxiv.org/abs/1409.3215 |
| bahdanau2015neural | forecasting_methods | Additive (Bahdanau) attention used by our decoder over the encoder history window. | https://arxiv.org/abs/1409.0473 |
| bengio2015scheduled | forecasting_methods | Scheduled sampling; justifies the annealed teacher-forcing curriculum used to train the autoregressive decoder. | https://arxiv.org/abs/1506.03099 |
| hyndman2006another | forecasting_methods | Scaled errors relative to the naive forecast; grounds our skill-over-persistence evaluation. | https://doi.org/10.1016/j.ijforecast.2006.03.001 |
| huber1964robust | forecasting_methods | Origin of the Huber loss used in the robust-training variant against heavy-tailed RTT/jitter spikes. | https://doi.org/10.1214/aoms/1177703732 |
| zeng2023transformers | forecasting_methods | Simple linear models rival Transformers; supports strong ridge/persistence baselines. | https://doi.org/10.1609/aaai.v37i9.26317 |
| lei2018distribution | uncertainty | Canonical split-conformal procedure used to calibrate per-horizon prediction intervals. | https://doi.org/10.1080/01621459.2017.1307116 |
| stankeviciute2021conformal | uncertainty | Conformalized multi-horizon RNN forecasting (CF-RNN); closest analogue to our conformalized Seq2Seq. | https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html |
| lakshminarayanan2017simple | uncertainty | Deep ensembles; justifies seed ensembles for accuracy and spread-based uncertainty. | https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html |
| gneiting2007strictly | uncertainty | Proper scoring rules incl. the interval (Winkler-type) score used with coverage and width. | https://doi.org/10.1198/016214506000001437 |
| barber2023conformal | uncertainty | Coverage-gap bounds under non-exchangeable data; explains deviation from nominal coverage on drifting link traces. | https://doi.org/10.1214/23-AOS2276 |
| gibbs2021adaptive | uncertainty | Adaptive conformal inference under distribution shift; the extension path for non-stationary handover regimes. | https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html |
| deconinck2017multipath | multipath_ntn | Defines MPQUIC and its default scheduler, which the prediction-aware scheduler extends. | https://doi.org/10.1145/3143361.3143370 |
| paasch2014experimental | multipath_ntn | Source of the minRTT (lowest-RTT-first) scheduler used as the reactive baseline. | https://doi.org/10.1145/2630088.2631977 |
| ferlin2016blest | multipath_ntn | HoL-blocking-aware scheduler for heterogeneous paths; baseline and motivation for predicted path RTT. | https://doi.org/10.1109/IFIPNetworking.2016.7497206 |
| lim2017ecf | multipath_ntn | Earliest-completion-first scheduler on current RTT/cwnd estimates; natural target for substituting forecasts. | https://doi.org/10.1145/3143361.3143376 |
| wu2020peekaboo | multipath_ntn | Online-RL multipath scheduler; contrasts learning-based with forecast-driven scheduling. | https://doi.org/10.1109/JSAC.2020.3000365 |
| yang2024mams | multipath_ntn | Mobility/handover-aware MPQUIC scheduler; closest prior prediction-aware scheduling work. | https://doi.org/10.1109/TNET.2024.3382269 |
| lin20215g | multipath_ntn | 3GPP Rel-17 NTN overview; standards context for LEO + 5G terrestrial integration. | https://doi.org/10.1109/MCOMSTD.011.2100038 |
| baltaci2023multipath | multipath_ntn | Measured MPTCP/MPQUIC over Starlink + cellular; shows scheduler failures on heterogeneous LEO/terrestrial paths. | https://doi.org/10.1109/ACCESS.2023.3325702 |

Count by topic: testbed 1, leo_measurement 9, link_prediction 7, forecasting_methods 7, uncertainty 6, multipath_ntn 8.

## Pool alternates (verified, not selected)

If the paper needs swaps or the 2 spare slots, these are the best candidates:

- **chi2026reading** (arXiv:2609.26696): concurrent LEO link-state forecaster with calibrated quantile bands. Cite it if reviewers ask about concurrent work. It is a preprint, not yet peer-reviewed.
- **lee2024handover**, **liu2024multiagent** (IEEE TWC 2024): learning-based LEO handover. Useful for fitting the target venue.
- **salinas2020deepar**, **lim2021temporal**, **oreshkin2020nbeats**, **nie2023time**: alternative deep forecasters.
- **xu2021conformal** (EnbPI), **angelopoulos2023conformal**, **vovk2005algorithmic**: extra conformal references.
- **ma2023network**, **bhosale2023astrolabe**, **laniewski2024wetlinks**, **ahangarpour2024trajectory**, **izhikevich2024democratizing**, **pan2024measuring**: extra LEO measurement references.
- **chen2025minimum**, **lanfer2024weather**, **casparsen2026statistical**, **mei2022realtime**, **minovski2023throughput**, **yue2018linkforecast**: extra prediction references.
- **yang2024mobility**, **zhu2022integrated**, **giordani2021nonterrestrial**, **azari2022evolution**, **xing2023online**: extra multipath/NTN references.

## Key map from the initial bibs

| the initial study key | New key |
|---|---|
| ISLAM2025111188 / islam_design_2024 | islam2025design |
| lumos5g | narayanan2020lumos5g |
| mams | yang2024mams |
| blest | ferlin2016blest |
| peek | wu2020peekaboo |
| leo_rtt_prediction | tian2025attention |
| 10723472 | basit2024predictability |
| 11310330 | perrin2025measurement |
| 10623111 | pan2024measuring |
| MEI2022108736 | mei2022realtime |
| kassem_browser-side_2022 | kassem2022browser |
| tanveer_making_2023 | tanveer2023making |
| michel_first_2022 | michel2022first |
| ma_network_2023 | ma2023network |
| izhikevich_democratizing_2024 | izhikevich2024democratizing |
