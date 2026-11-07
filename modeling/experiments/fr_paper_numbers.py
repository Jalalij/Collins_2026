"""
experiments/fr_paper_numbers.py - Every number of the journal paper that comes from the new analyses.

Writes LaTeX macros and two tables, each traced to a result file (the comment next to a macro
names its source). Run it after the analyses (fr_volatility, fr_condcov, fr_bootstrap,
fr_scheduler, fr_attention, fr_leak, fr_trace) and after final_aggregate.py.

Writes <paper>/figs/data/numbers_story.tex, tab_main_boot.tex, tab_extent.tex and the same
macros to <report>/data/numbers_story.tex, plus results_final/analysis/paper_numbers.json.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_final"
AN = RES / "analysis"
LINKS = ["oneweb_dl", "oneweb_ul", "starlink_dl", "starlink_ul", "quectel_dl", "quectel_ul"]
NAMES = {"oneweb_dl": "OneWeb DL", "oneweb_ul": "OneWeb UL", "starlink_dl": "Starlink DL",
         "starlink_ul": "Starlink UL", "quectel_dl": "5G DL", "quectel_ul": "5G UL"}
TAG = {"oneweb_dl": "OWDL", "oneweb_ul": "OWUL", "starlink_dl": "SLDL", "starlink_ul": "SLUL",
       "quectel_dl": "QDL", "quectel_ul": "QUL"}

MAC, SRC = {}, {}


def signs(t):
    """Typeset signs of numbers in running text: -1.6 -> $-$1.6, +0.4 -> $+$0.4."""
    return re.sub(r"(?<![\w$\\])([+-])(?=\d)", lambda g: "$-$" if g.group(1) == "-" else "$+$", t)


def m(name, value, src, fmt="{:.1f}"):
    """Store one macro. value may be a number (formatted with fmt) or a ready string."""
    MAC[name] = signs(value if isinstance(value, str) else fmt.format(value))
    SRC[name] = src


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=str(ROOT.parent / "paper_twc"))
    ap.add_argument("--report", default=str(ROOT.parent / "reports/final_report"))
    a = ap.parse_args()
    paper, report = Path(a.paper), Path(a.report)

    # ------------------------------------------------------------------ data set
    ds = json.load(open(AN / "dataset_stats.json"))
    src = "results_final/analysis/dataset_stats.json"
    m("DataSessions", f"{ds['total']['files']:,}".replace(",", "{,}"), src)
    m("DataRows", f"{ds['total']['rows'] / 1e6:.2f}", src + " (million rows)")
    m("DataHours", f"{ds['total']['hours']:.0f}", src)
    m("DataDaysOW", str(ds["oneweb_dl"]["days"]), src)
    m("DataDaysSL", str(ds["starlink_dl"]["days"]), src)
    m("DataDaysQ", str(ds["quectel_dl"]["days"]), src)
    for ln in LINKS:
        m("DataHours" + TAG[ln], ds[ln]["hours"], src)

    # ------------------------------------------------------------------ summary (all methods)
    sm = pd.read_csv(RES / "summary/summary.csv").set_index(["dataset", "method"])
    src = "results_final/summary/summary.csv"
    rm = []
    for ln in LINKS:
        ref = sm.loc[(ln, "prev_seq2seq_ens"), "rtt_RMSE"]
        for meth in ["ridge_fr", "lqr_fr", "fr_resid_ens", "fr_full_ens", "fr_dist_ens", "fr_gauss_ens", "fr_dist_w4_ens",
                     "fr_resid", "fr_full", "fr_dist", "fr_gauss", "fr_dist_w4"]:
            if (ln, meth) in sm.index:
                rm.append(-pct(sm.loc[(ln, meth), "rtt_RMSE"], ref))
    m("MaxRMSEGain", float(np.ceil(max(rm))), src + " (largest RMSE reduction vs S2S-E, any model, any link)", "{:.0f}")
    ratio = [sm.loc[(ln, "persistence"), "rtt_MAE"] / sm.loc[(ln, "fr_dist_ens"), "rtt_MAE"] for ln in LINKS]
    m("PersRatioLo", min(ratio), src); m("PersRatioHi", max(ratio), src)
    iso = [abs(pct(sm.loc[(ln, "fr_dist_ens"), "rtt_MIS_y2style"], sm.loc[(ln, "prev_seq2seq_ens"), "rtt_MIS_y2style"]))
           for ln in LINKS]
    m("IsProtoMaxAbs", float(np.ceil(max(iso) * 10) / 10), src + " rtt_MIS_y2style (test-residual intervals)")

    # ------------------------------------------------------------------ volatility (Section IV)
    acf = pd.read_csv(AN / "volatility/acf_summary.csv").set_index("link")
    wc = pd.read_csv(AN / "volatility/window_corr.csv").set_index("link")
    src = "results_final/analysis/volatility/acf_summary.csv"
    five = [ln for ln in LINKS if ln != "starlink_dl"]
    m("VolDmaxAbs", float(np.ceil(acf.loc[five, "d_max_abs_lag0p4to10s"].max() * 100) / 100), src + " d_max_abs_lag0p4to10s", "{:.2f}")
    ow5 = ["oneweb_dl", "oneweb_ul", "quectel_dl"]
    m("VolAbsdLo", acf.loc[ow5, "absd_1s_measured_within"].min(), src + " absd_1s_measured_within", "{:.2f}")
    m("VolAbsdHi", acf.loc[ow5, "absd_1s_measured_within"].max(), src, "{:.2f}")
    m("VolSLDLlevel", acf.loc["starlink_dl", "rtt_1s_measured_within"], src + " rtt_1s_measured_within", "{:.2f}")
    m("VolSLDLlevelMin", float(np.floor(acf.loc["starlink_dl", "rtt_1s_sessions_min"] * 100) / 100), src + " rtt_1s_sessions_min", "{:.2f}")
    m("VolSLDLnSess", str(int(acf.loc["starlink_dl", "n_train_sessions"])), src)
    src = "results_final/analysis/volatility/window_corr.csv"
    m("VolSignRevLo", 100 * wc["sign_revert_to_input_mean_balanced"].min(), src, "{:.0f}")
    m("VolSignRevHi", 100 * wc["sign_revert_to_input_mean_balanced"].max(), src, "{:.0f}")
    nn = pd.concat([wc["sign_s2s_e_balanced"], wc["sign_ds2s_e_balanced"]])
    m("VolSignNNLo", 100 * nn.min(), src, "{:.0f}"); m("VolSignNNHi", 100 * nn.max(), src, "{:.0f}")
    owq = ["oneweb_dl", "oneweb_ul", "quectel_dl", "quectel_ul"]
    m("VolWithinLo", wc.loc[owq, "corr_inputsd_abs_err_s2s_e_within"].min(), src + " corr_inputsd_abs_err_s2s_e_within", "{:.2f}")
    m("VolWithinHi", wc.loc[owq, "corr_inputsd_abs_err_s2s_e_within"].max(), src, "{:.2f}")
    m("VolBetweenLo", wc.loc[owq, "corr_inputsd_abs_err_s2s_e_between"].min(), src + " corr_inputsd_abs_err_s2s_e_between", "{:.2f}")
    m("VolBetweenHi", wc.loc[owq, "corr_inputsd_abs_err_s2s_e_between"].max(), src, "{:.2f}")
    m("VolSLDLrv", wc.loc["starlink_dl", "corr_input_rv_horizon_rv_within"], src + " corr_input_rv_horizon_rv_within", "{:.2f}")

    # skew of the S2S-E 1 s error on the calibration sessions (h = 5)
    def ens(ln, meth, key="p_va"):
        fs = [RES / f"main/{ln}/preds/{meth}_s{k}.npz" for k in range(3)]
        fs = [f for f in fs if f.exists()]
        return np.mean([np.load(f)[key][:, :, 0] for f in fs], 0)
    for ln in ["quectel_ul", "oneweb_dl"]:
        y = np.load(RES / f"main/{ln}/preds/truth.npz")["y_va"][:, 4, 0]
        r = y - ens(ln, "prev_seq2seq")[:, 4]
        m("Skew" + TAG[ln], np.quantile(r, 0.975) / abs(np.quantile(r, 0.025)),
          f"results_final/main/{ln}/preds (S2S-E calibration-session error at 1 s, q.975 / |q.025|)")

    # ------------------------------------------------------------------ conditional coverage
    cc = pd.read_csv(AN / "condcov/summary.csv").set_index(["link", "method"])
    ccj = json.load(open(AN / "condcov/summary.json"))
    src = "results_final/analysis/condcov/summary.csv"
    c1 = [cc.loc[(ln, "s2s_const"), "cov_d1"] for ln in LINKS]
    c10 = [cc.loc[(ln, "s2s_const"), "cov_d10"] for ln in LINKS]
    m("CovConstCalmLo", min(c1), src, "{:.3f}"); m("CovConstCalmHi", max(c1), src, "{:.3f}")
    m("CovConstVolLo", min(c10), src, "{:.3f}"); m("CovConstVolHi", max(c10), src, "{:.3f}")
    assert all(x > y for x, y in zip(c1, c10)), "calm > volatile does not hold on every link"
    for ln in ["starlink_dl", "quectel_ul"]:
        m(f"CovConst{TAG[ln]}dOne", cc.loc[(ln, "s2s_const"), "cov_d1"], src, "{:.3f}")
        m(f"CovConst{TAG[ln]}dTen", cc.loc[(ln, "s2s_const"), "cov_d10"], src, "{:.3f}")
    m("CovNormOWDLdOne", cc.loc[("oneweb_dl", "s2s_norm"), "cov_d1"], src, "{:.3f}")
    for ln in ["starlink_dl", "quectel_dl", "quectel_ul"]:
        m("GapDS" + TAG[ln], cc.loc[(ln, "ds2s_cqr"), "maxdev"], src + " maxdev", "{:.3f}")
        m("GapConst" + TAG[ln], cc.loc[(ln, "s2s_const"), "maxdev"], src, "{:.3f}")
    for ln in ["oneweb_dl", "oneweb_ul"]:
        m("WidRatio" + TAG[ln], cc.loc[(ln, "ds2s_cqr"), "wid_ratio_d10_d1"], src + " wid_ratio_d10_d1", "{:.2f}")
        m("CovDS" + TAG[ln] + "dTen", cc.loc[(ln, "ds2s_cqr"), "cov_d10"], src, "{:.3f}")
        m("CovNormTuned" + TAG[ln] + "dTen", cc.loc[(ln, "s2s_normtuned"), "cov_d10"], src, "{:.3f}")
        w = pd.read_csv(report / f"data/fr_condcov_wid_{ln}.dat", sep=" ")
        m("ErrRatio" + TAG[ln], w.wids2smondrian.iloc[-1] / w.wids2smondrian.iloc[0],
          f"reports/final_report/data/fr_condcov_wid_{ln}.dat (per-decile calibrated width at 1 s, d10/d1)")
    w = pd.read_csv(report / "data/fr_condcov_wid_starlink_dl.dat", sep=" ")
    m("WidConstSLDL", w.wids2sconst.iloc[0], "reports/final_report/data/fr_condcov_wid_starlink_dl.dat", "{:.0f}")
    m("WidDSSLDLcalmLo", w.widds2scqr.iloc[:3].min(), "fr_condcov_wid_starlink_dl.dat deciles 1-3", "{:.0f}")
    m("WidDSSLDLcalmHi", w.widds2scqr.iloc[:3].max(), "fr_condcov_wid_starlink_dl.dat deciles 1-3", "{:.0f}")
    m("WidDSSLDLloadLo", w.widds2scqr.iloc[3:].min(), "fr_condcov_wid_starlink_dl.dat deciles 4-10", "{:.0f}")
    m("WidDSSLDLloadHi", w.widds2scqr.iloc[3:].max(), "fr_condcov_wid_starlink_dl.dat deciles 4-10", "{:.0f}")
    # decomposition: centre, width, asymmetry
    cen = [pct(cc.loc[(ln, "ds2s_meanconst"), "mis"], cc.loc[(ln, "s2s_const"), "mis"]) for ln in LINKS]
    wid = [pct(cc.loc[(ln, "s2s_ds2swidth"), "mis"], cc.loc[(ln, "s2s_const"), "mis"]) for ln in LINKS]
    asy = [pct(cc.loc[(ln, "ds2s_cqr"), "mis"], cc.loc[(ln, "s2s_ds2swidth"), "mis"]) for ln in LINKS]
    m("DecCentreLo", min(cen), src + " ds2s_meanconst vs s2s_const"); m("DecCentreHi", max(cen), src)
    m("DecWidthLoAbs", -max(wid), src + " s2s_ds2swidth vs s2s_const"); m("DecWidthHiAbs", -min(wid), src)
    m("DecAsymLoAbs", -max(asy), src + " ds2s_cqr vs s2s_ds2swidth"); m("DecAsymHiAbs", -min(asy), src)
    m("AsymQULratio", ccj["links"]["quectel_ul"]["info"]["ds2s_cqr"]["asym_h5"]["upper_over_lower_median"],
      "results_final/analysis/condcov/summary.json info.ds2s_cqr.asym_h5", "{:.0f}")

    # extent table and its macros
    rows = [("S2S-E", "constant", "s2s_const"), ("S2S-E", "asymmetric", "s2s_asym"),
            ("S2S-E", "SD-scaled", "s2s_normtuned"), ("S2S-AE-E", "constant", "frfull_const"),
            ("LQR", "CQR", "lqr_cqr"), ("GS2S-E", "CQR", "gauss_cqr"), ("DS2S-E", "CQR", "ds2s_cqr")]
    have = {k for (_, k) in cc.index}
    rows = [r for r in rows if r[2] in have]
    lines = []
    for fc, iv, k in rows:
        cells = []
        for ln in LINKS:
            if (ln, k) not in cc.index:
                cells += ["--", "--"]; continue
            v = cc.loc[(ln, k), "mis"]
            best = min(cc.loc[(ln, kk), "mis"] for (_, _, kk) in rows if (ln, kk) in cc.index)
            t = f"{v:.1f}" if v < 1000 else f"{v:.0f}"
            cells += [r"\textbf{" + t + "}" if np.isclose(v, best) else t, f"{cc.loc[(ln, k), 'maxdev']:.3f}"]
        lines.append(f"{fc} & {iv} & " + " & ".join(cells) + r" \\")
        if k in ("s2s_normtuned", "frfull_const"):
            lines.append(r"\midrule")
    (report / "data/tab_extent.tex").write_text("% generated by modeling/experiments/fr_paper_numbers.py\n" + "\n".join(lines) + "\n")
    (paper / "figs/data/tab_extent.tex").write_text("% generated by modeling/experiments/fr_paper_numbers.py from "
                                                    "results_final/analysis/condcov/summary.csv\n" + "\n".join(lines) + "\n")
    bestpt = [min(cc.loc[(ln, k), "mis"] for k in ("s2s_const", "s2s_asym", "s2s_normtuned") if (ln, k) in cc.index)
              for ln in LINKS]
    gap = [pct(b, cc.loc[(ln, "ds2s_cqr"), "mis"]) for ln, b in zip(LINKS, bestpt)]
    m("ExtBestPointLoAbs", min(gap), src + " best point-forecast interval vs ds2s_cqr", "{:.0f}")
    m("ExtBestPointHiAbs", max(gap), src, "{:.0f}")
    if all((ln, "frfull_const") in cc.index for ln in LINKS):
        ff = [pct(cc.loc[(ln, "frfull_const"), "mis"], cc.loc[(ln, "s2s_const"), "mis"]) for ln in LINKS]
        m("ExtFrfullLo", min(ff), src + " frfull_const vs s2s_const"); m("ExtFrfullHi", max(ff), src)
    lq = [abs(pct(cc.loc[(ln, "lqr_cqr"), "mis"], cc.loc[(ln, "ds2s_cqr"), "mis"])) for ln in LINKS]
    m("ExtLQRMaxAbs", max(lq), src + " |lqr_cqr vs ds2s_cqr|", "{:.1f}")
    # Gaussian heads
    gl = [ln for ln in LINKS if (ln, "gauss_cqr") in cc.index]
    gseeds = sorted({len(ccj["links"][ln]["info"]["gauss_cqr"]["seeds"]) for ln in gl}) if gl else []
    if len(gl) == 6:
        gg = [pct(cc.loc[(ln, "gauss_cqr"), "mis"], cc.loc[(ln, "ds2s_cqr"), "mis"]) for ln in LINKS]
        worse = sum(g > 0 for g in gg)
        worst = LINKS[int(np.argmax(gg))]
        if worse >= 4:
            head = r"Third, Gaussian heads do not recover the gain."
        elif worse >= 1:
            head = r"Third, Gaussian heads recover only part of the gain."
        else:
            head = r"Third, Gaussian heads do as well as the free quantiles."
        m("GaussSentence", (head + r" GS2S-E, whose quantiles are symmetric, has a higher interval score than DS2S-E on %d of the "
                            r"six links (%+.1f\%% to %+.1f\%%), with the largest gap on %s." % (worse, min(gg), max(gg), NAMES[worst])
                            ).replace("+-", "-"), src + " gauss_cqr vs ds2s_cqr")
        m("GaussSeedNote", "uses %s seed%s" % ("/".join(map(str, gseeds)), "" if gseeds == [1] else "s"), src)
    else:
        m("GaussSentence", r"Third, Gaussian heads are still training and are left out here.", "pending")
        m("GaussSeedNote", "pending", "pending")

    # ------------------------------------------------------------------ bootstrap (Table II, validity)
    bs = pd.read_csv(AN / "bootstrap/bootstrap_summary.csv")
    src = "results_final/analysis/bootstrap/bootstrap_summary.csv"
    main = bs[(bs.method == "fr_dist") & (bs.role == "main")]
    def row(ln, met):
        r = main[(main.link == ln) & (main.metric == met)]
        return r.iloc[0]
    isv = [row(ln, "IS").change_full_pct for ln in LINKS]
    m("BSISWorstAbs", -max(isv), src, "{:.1f}"); m("BSISBestAbs", -min(isv), src, "{:.1f}")
    m("BSBestMAEAbs", -min(row(ln, "MAE").change_full_pct for ln in LINKS), src, "{:.1f}")
    sigmae = [row(ln, "MAE").change_full_pct for ln in LINKS if row(ln, "MAE").boot_hi_pct < 0]
    m("BSMAEsigLoAbs", -max(sigmae), src, "{:.1f}"); m("BSMAEsigHiAbs", -min(sigmae), src, "{:.1f}")
    rmse = [row(ln, "RMSE").change_full_pct for ln in LINKS]
    m("BSRMSEMaxAbs", float(np.ceil(max(abs(x) for x in rmse) * 10) / 10), src, "{:.1f}")
    m("BSnSessTot", str(int(sum(row(ln, "IS").n_sessions for ln in LINKS))), src)
    assert all(row(ln, "IS").sessions_better == row(ln, "IS").n_sessions for ln in LINKS), "IS not better in every session"
    m("BSWidRatioLo", main.width_ratio_session_vs_naive.min(), src + " width_ratio_session_vs_naive (main rows)")
    m("BSWidRatioHi", main.width_ratio_session_vs_naive.max(), src)
    loo = [max(abs(row(ln, "IS").loo_min_pct - row(ln, "IS").change_full_pct),
               abs(row(ln, "IS").loo_max_pct - row(ln, "IS").change_full_pct)) for ln in LINKS]
    m("BSLooMax", max(loo), src + " loo_min_pct / loo_max_pct (IS)")
    nwin = sum(np.load(RES / f"main/{ln}/preds/truth.npz")["y_te"].shape[0] for ln in LINKS)
    m("BSnWinTot", f"{nwin:,}".replace(",", "{,}"), "results_final/main/*/preds/truth.npz")
    pj = pd.read_csv(AN / "bootstrap/point_and_jumps.csv")
    # MAE of the DS2S-E mean and median against the S2S-E mean (full test set)
    mm, md = {}, {}
    for ln in LINKS:
        y = np.load(RES / f"main/{ln}/preds/truth.npz")["y_te"][:, :, 0]
        s = ens(ln, "prev_seq2seq", "p_te")
        dmean = ens(ln, "fr_dist", "p_te")
        fs = [RES / f"main/{ln}/preds/fr_dist_s{k}.npz" for k in range(3)]
        dmed = np.mean([np.load(f)["d_te"][:, :, 0, 3].astype(np.float64) for f in fs], 0)
        ref = np.abs(y - s).mean()
        mm[ln], md[ln] = pct(np.abs(y - dmean).mean(), ref), pct(np.abs(y - dmed).mean(), ref)
    o5 = [ln for ln in LINKS if ln != "starlink_dl"]
    m("MeanMAELo", min(mm[ln] for ln in o5), "preds npz (MAE of the DS2S-E mean vs S2S-E)")
    m("MeanMAEHi", max(mm[ln] for ln in o5), "preds npz")
    m("MedMAELo", min(md[ln] for ln in o5), "preds npz (MAE of the DS2S-E median vs S2S-E)")
    m("MedMAEHi", max(md[ln] for ln in o5), "preds npz")
    m("MeanMAESLDL", mm["starlink_dl"], "preds npz")
    # Table II rows
    lines = []
    for ln in LINKS:
        cells = [NAMES[ln]]
        for met, fmt in (("IS", "{:.1f}"), ("MAE", "{:.2f}"), ("RMSE", "{:.2f}")):
            r = row(ln, met)
            sig = r.boot_hi_pct < 0 or r.boot_lo_pct > 0
            ch = f"{r.change_full_pct:+.1f} [{r.boot_lo_pct:+.1f}, {r.boot_hi_pct:+.1f}]".replace("+-", "-")
            ch = ch.replace("-", "$-$").replace("+", "$+$")
            v1, v0 = r.method_value, r.base_value
            f1 = fmt.format(v1) if v1 < 1000 else f"{v1:.0f}"
            f0 = fmt.format(v0) if v0 < 1000 else f"{v0:.0f}"
            cells += [f0, f1, r"\textbf{" + ch + "}" if sig else ch]
        cells += [f"{sm.loc[(ln, 'prev_seq2seq_ens'), 'rtt_PICP']:.3f}", f"{sm.loc[(ln, 'fr_dist_ens'), 'rtt_PICP']:.3f}"]
        lines.append(" & ".join(cells) + r" \\")
    (report / "data/tab_main_boot.tex").write_text("% generated by modeling/experiments/fr_paper_numbers.py\n" + "\n".join(lines) + "\n")
    (paper / "figs/data/tab_main_boot.tex").write_text("% generated by modeling/experiments/fr_paper_numbers.py from "
                                                       "results_final/analysis/bootstrap/bootstrap_summary.csv\n"
                                                       + "\n".join(lines) + "\n")
    # DS2S-E vs LQR (paired session bootstrap)
    mv = pd.read_csv(AN / "bootstrap/main_vs_others.csv")
    mv = mv[(mv.method == "fr_dist") & (mv.against == "lqr_fr")]
    src = "results_final/analysis/bootstrap/main_vs_others.csv (against lqr_fr)"
    r_ = mv[mv.metric == "RMSE"].set_index("link")
    assert (r_.boot_hi_pct < 0).all(), "DS2S-E RMSE not significantly below LQR on every link"
    m("LQRRMSELoAbs", -r_.change_full_pct.max(), src); m("LQRRMSEHiAbs", -r_.change_full_pct.min(), src)
    i_ = mv[mv.metric == "IS"].set_index("link")
    sig = i_[i_.boot_hi_pct < 0]
    m("LQRsigLo", -sig.change_full_pct.max(), src, "{:.0f}"); m("LQRsigHi", -sig.change_full_pct.min(), src, "{:.0f}")
    m("LQRsigLinks", ", ".join(NAMES[ln] for ln in sig.index), src)

    # ------------------------------------------------------------------ leak
    lk = json.load(open(AN / "leak/leak_summary.json"))
    src = "results_final/analysis/leak/leak_summary.json"
    m("LeakOptDSMAEhOne", lk["DS2S_MAE"]["optimism_h1_pct"], src, "{:.0f}")
    m("LeakOptDSMAEavg", lk["DS2S_MAE"]["optimism_avg_pct"], src)
    m("LeakOptDSRMSEhOne", lk["DS2S_RMSE"]["optimism_h1_pct"], src, "{:.0f}")
    m("LeakOptSSMAEhOne", lk["S2S_MAE"]["optimism_h1_pct"], src)
    m("LeakDSAhOne", lk["DS2S_MAE"]["A_h1"], src); m("LeakDSBhOne", lk["DS2S_MAE"]["B_h1"], src)
    m("LeakDSChOne", lk["DS2S_MAE"]["C_h1"], src)

    # ------------------------------------------------------------------ trace
    tr = json.load(open(AN / "trace/trace_starlink_dl.json"))
    src = "results_final/analysis/trace/trace_starlink_dl.json"
    m("TraceWidS", tr["s2s_width_ms"], src, "{:.0f}")
    m("TraceWidDCalm", tr["ds2s_width_calm_median_ms"], src, "{:.0f}")
    m("TraceWidDLoad", tr["ds2s_width_load_median_ms"], src, "{:.0f}")

    # ------------------------------------------------------------------ scheduler
    hd = pd.read_csv(AN / "scheduler/headline.csv")
    h = hd[(hd.deadline == "D90") & (hd.ref_bound == "one95")].set_index("link")
    src = "results_final/analysis/scheduler/headline.csv (D90, one95)"
    sl = h.loc["starlink_dl"]
    m("SchedDSLDL", sl.D_ms, src, "{:.0f}")
    m("SchedSSdupSLDL", 100 * sl.s2s_dup, src); m("SchedSSmissSLDL", 100 * sl.s2s_miss, src, "{:.2f}")
    m("SchedDSdupSLDL", 100 * sl.valmatch_ds2s_dup, src); m("SchedDSmissSLDL", 100 * sl.valmatch_ds2s_miss, src, "{:.2f}")
    assert sl.valmatch_ds2s_miss <= sl.s2s_miss
    m("SchedSaveSLDL", 100 * sl.valmatch_saving_vs_s2s_point, src, "{:.0f}")
    m("SchedSaveSLDLlo", 100 * sl.valmatch_saving_vs_s2s_point_lo, src, "{:.0f}")
    m("SchedSaveSLDLhi", 100 * sl.valmatch_saving_vs_s2s_point_hi, src, "{:.0f}")
    for ln in ["starlink_ul", "oneweb_ul"]:
        m("SchedSave" + TAG[ln], -100 * h.loc[ln].valmatch_saving_vs_s2s_point, src, "{:+.1f}")
    ow = h.loc["oneweb_dl"]
    m("SchedDSdupOWDL", 100 * ow.valmatch_ds2s_dup, src); m("SchedSSdupOWDL", 100 * ow.s2s_dup, src)
    m("SchedDSdupCurveOWDL", 100 * ow.ds2s_curve_dup_at_s2s_miss, src)
    m("SchedDSmissOWDL", 100 * ow.valmatch_ds2s_miss, src, "{:.2f}"); m("SchedSSmissOWDL", 100 * ow.s2s_miss, src, "{:.2f}")
    m("SchedFFdupSLDL", 100 * sl.frfull_curve_dup_at_s2s_miss, src)
    m("SchedDSdupCurveSLDL", 100 * sl.ds2s_curve_dup_at_s2s_miss, src)
    m("SchedSSdupCurveSLDL", 100 * sl.s2s_curve_dup_at_s2s_miss, src)
    m("SchedSaveFFSLDL", 100 * sl.valmatch_saving_vs_frfull, src, "{:.0f}")
    lj = json.load(open(AN / "scheduler/link_starlink_dl.json"))
    dec = lj.get("deadlines", {}).get("D90", {}).get("decomposition_at_s2s95_miss", {})
    ns = lj.get("n_seeds", {})
    if "gauss" in dec:
        lab = "GS2S-E" if ns.get("gauss", 0) >= 3 else "GS2S (%d seed%s)" % (ns.get("gauss", 1), "" if ns.get("gauss", 1) == 1 else "s")
        g, ff, dd = dec["gauss"], dec["frfull"], dec["ds2s"]
        tail = ("close to the point forecaster S2S-AE-E, so a symmetric spread adds little" if abs(g - ff) < abs(g - dd)
                else "between S2S-AE-E and DS2S-E, so a symmetric spread recovers part of the gain")
        m("GaussSchedSentence", (r"%s, whose quantiles are symmetric, needs %.1f\%% at the same miss rate, %s." % (lab, 100 * g, tail)),
          "results_final/analysis/scheduler/link_starlink_dl.json decomposition_at_s2s95_miss.gauss")
    else:
        m("GaussSchedSentence", "", "no Gaussian curve")
    bc = pd.read_csv(AN / "scheduler/backup_check.csv")
    src = "results_final/analysis/scheduler/backup_check.csv"
    dl = bc[(bc.backup == "quectel_dl") & bc.paired]
    m("BackupDLmax", float(np.ceil(100 * dl.backup_frac_above_D.max() * 100) / 100), src, "{:.2f}")
    su = bc[(bc.primary == "starlink_ul") & (bc.backup == "quectel_ul") & (bc.deadline == "D90")].iloc[0]
    m("BackupQULSL", 100 * su.backup_frac_above_D, src)
    m("BackupSSmissSLUL", 100 * su.s2s95_miss, src, "{:.2f}")
    m("BackupSSmissCorrSLUL", 100 * su.s2s95_miss_corrected, src, "{:.2f}")
    m("BackupDSmissSLUL", 100 * su.ds2s95_miss, src, "{:.2f}")
    m("BackupDSmissCorrSLUL", 100 * su.ds2s95_miss_corrected, src, "{:.2f}")
    rk = pd.read_csv(AN / "scheduler/riskmap_starlink_dl.csv")
    b5 = rk[rk.mean_bin == 5]
    src = "results_final/analysis/scheduler/riskmap_starlink_dl.csv (mean decile 5)"
    m("RiskMeanLo", b5.s2s_mean_ms.min(), src, "{:.0f}"); m("RiskMeanHi", b5.s2s_mean_ms.max(), src, "{:.0f}")
    m("RiskLowPct", 100 * b5[b5.spread_q == b5.spread_q.min()].rate.iloc[0], src)
    m("RiskHighPct", 100 * b5[b5.spread_q == b5.spread_q.max()].rate.iloc[0], src)
    cov = []
    for ln in ["starlink_dl", "starlink_ul", "oneweb_dl", "oneweb_ul"]:
        c = pd.read_csv(AN / f"scheduler/calibrated_{ln}.csv")
        c = c[(c.deadline == "D90") & (c.policy.isin(["s2s", "ds2s"])) & (c.level.astype(str) == "one0.95")]
        if "coverage_of_U" in c:
            cov += list(c.coverage_of_U)
    if cov:
        m("BoundCovLo", min(cov), "results_final/analysis/scheduler/calibrated_<link>.csv coverage_of_U", "{:.3f}")
        m("BoundCovHi", max(cov), "results_final/analysis/scheduler/calibrated_<link>.csv coverage_of_U", "{:.3f}")

    # ------------------------------------------------------------------ attention
    at = pd.read_csv(AN / "attention/attn_summary.csv")
    at = at[at.model == "fr_dist"].set_index("link")
    src = "results_final/analysis/attention/attn_summary.csv"
    m("GradLastLo", 100 * at.grad_sum_change_mass_last1s.min(), src + " grad_sum_change_mass_last1s", "{:.0f}")
    m("GradLastHi", 100 * at.grad_sum_change_mass_last1s.max(), src, "{:.0f}")
    sq = ["starlink_dl", "starlink_ul", "quectel_dl", "quectel_ul"]
    m("AttnAblMax", 100 * (at.loc[sq, ["abl_mae_ratio_uniform", "abl_mae_ratio_valmap"]].max().max() - 1), src, "{:.1f}")
    m("AttnAblOWDL", 100 * (at.loc["oneweb_dl", "abl_mae_ratio_uniform"] - 1), src, "{:.0f}")
    m("AttnAblOWULmap", 100 * (at.loc["oneweb_ul", "abl_mae_ratio_valmap"] - 1), src, "{:.0f}")
    aj = json.load(open(AN / "attention/attn_summary.json"))["links"]
    sink = {ln: aj[ln]["fr_dist"]["profile_over_steps_x30"][-1] / 30 for ln in LINKS}
    m("AttnSinkLo", 100 * min(sink[ln] for ln in ["oneweb_dl", "oneweb_ul", "quectel_ul"]), "attn_summary.json profile_over_steps_x30[29]/30", "{:.0f}")
    m("AttnSinkHi", 100 * max(sink[ln] for ln in ["oneweb_dl", "oneweb_ul", "quectel_ul"]), "attn_summary.json", "{:.0f}")
    m("GradEngLo", 100 * at.grad_h15_eng_feature_share.min(), src + " grad_h15_eng_feature_share", "{:.0f}")
    m("GradEngHi", 100 * at.grad_h15_eng_feature_share.max(), src, "{:.0f}")

    # ------------------------------------------------------------------ sweeps and pinball weight
    sw = pd.read_csv(RES / "summary/sweeps.csv")
    src = "results_final/summary/sweeps.csv"
    ch = {"MIS": [], "MAE": [], "RMSE": []}
    for cfg in ["base", "L15", "L60", "H30"]:
        for ln in ["oneweb_dl", "oneweb_ul", "starlink_dl", "starlink_ul"]:
            s = sw[(sw.cfg == cfg) & (sw.dataset == ln)].set_index("method")
            if "fr_dist" in s.index and "prev_seq2seq" in s.index:
                for met in ch:
                    ch[met].append(pct(s.loc["fr_dist", met], s.loc["prev_seq2seq", met]))
    m("SweepISLoAbs", -max(ch["MIS"]), src, "{:.0f}"); m("SweepISHiAbs", -min(ch["MIS"]), src, "{:.0f}")
    assert max(ch["MAE"]) < 0
    m("SweepRMSELo", min(ch["RMSE"]), src); m("SweepRMSEHi", max(ch["RMSE"]), src)
    # pinball weight 4 in the main protocol (seed means), all links with >= 2 seeds
    hz = RES / "summary/runs.csv"
    rs = pd.read_csv(hz) if hz.exists() else None
    wsent = ""
    if rs is not None and "fr_dist_w4" in set(rs.method):
        got = []
        for ln in LINKS:
            a4 = rs[(rs.dataset == ln) & (rs.method == "fr_dist_w4")]
            a1 = rs[(rs.dataset == ln) & (rs.method == "fr_dist")]
            if a4.seed.nunique() >= 2 and len(a1):
                got.append((ln, a4.seed.nunique(), pct(a4.rtt_MAE.mean(), a1.rtt_MAE.mean()),
                            pct(a4.rtt_MIS.mean(), a1.rtt_MIS.mean()), pct(a4.rtt_RMSE.mean(), a1.rtt_RMSE.mean())))
        if got:
            # model selection on the CALIBRATION sessions: mean pinball loss of the RTT quantiles (seed mean)
            tau = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
            def val_pinball(ln, meth):
                y = np.load(RES / f"main/{ln}/preds/truth.npz")["y_va"][:, :, 0]
                out = []
                for k in range(3):
                    f = RES / f"main/{ln}/preds/{meth}_s{k}.npz"
                    if f.exists():
                        q = np.load(f)["d_va"][:, :, 0, 1:].astype(np.float64)
                        e = y[..., None] - q
                        out.append(np.maximum(tau * e, (tau - 1) * e).mean())
                return np.mean(out)
            pref = [ln for (ln, *_r) in got if val_pinball(ln, "fr_dist_w4") < val_pinball(ln, "fr_dist")]
            mae = [g[2] for g in got]; isc = [g[3] for g in got]; rmse_ = [g[4] for g in got]
            ns = sorted({g[1] for g in got})
            nw = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
            pref_txt = ("all %s" % nw.get(len(got), str(len(got)))) if len(pref) == len(got) else "%s of the %s" % (nw.get(len(pref), str(len(pref))), nw.get(len(got), str(len(got))))
            wsent = (r"A pinball weight $\omega=4$, which the calibration sessions prefer on %s links, "
                     r"changes the MAE of DS2S by %+.1f\%% to %+.1f\%%, the interval score by %+.1f\%% to %+.1f\%%, and the RMSE by "
                     r"%+.1f\%% to %+.1f\%% (means of %s seeds)."
                     % (pref_txt, min(mae), max(mae),
                        min(isc), max(isc), min(rmse_), max(rmse_), " or ".join(map(str, ns)))).replace("+-", "-")
            m("WeightPrefLinks", ", ".join(NAMES[ln] for ln in pref), "preds npz (calibration-session pinball loss)")
    m("WeightSentence", wsent, "results_final/summary/runs.csv (fr_dist_w4 vs fr_dist)" if wsent else "pending")
    js = report / "data/numbers_results.tex"
    txt = js.read_text() if js.exists() else ""
    g = lambda k: float(re.search(r"\\newcommand\{\\%s\}\{([^}]*)\}" % k, txt).group(1))
    m("JitISRatio", g("JitISSLDLS") / g("JitISSLDLDS"), "reports/final_report/data/numbers_results.tex JitIS*")

    # ------------------------------------------------------------------ write
    out = ["% Generated by modeling/experiments/fr_paper_numbers.py. Do not edit. Source of each macro in the comment."]
    for k in MAC:
        out.append(r"\newcommand{\%s}{%s}%% %s" % (k, MAC[k], SRC[k]))
    txt = "\n".join(out) + "\n"
    (paper / "figs/data/numbers_story.tex").write_text(txt)
    (report / "data/numbers_story.tex").write_text(txt)
    json.dump({k: {"value": MAC[k], "source": SRC[k]} for k in MAC}, open(AN / "paper_numbers.json", "w"), indent=1)
    print(f"{len(MAC)} macros written")
    for k in MAC:
        print(f"  {k} = {MAC[k]}")


if __name__ == "__main__":
    main()
