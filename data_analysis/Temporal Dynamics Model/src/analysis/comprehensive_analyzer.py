"""
Comprehensive analyzer module - coordinates all model training and analysis.
This is the main coordinator module that orchestrates the entire analysis pipeline.
"""

import os
import json
import warnings
from datetime import datetime
from typing import Dict, Any

warnings.filterwarnings("ignore")

import numpy as np

# Optional: Ljung–Box (statsmodels). Graceful fallback if not available.
try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:
    _HAS_SM = False

from scipy.stats import pearsonr

from ..evaluation.standardized_evaluation import (
    StandardizedModelEvaluator,
    STANDARD_TARGET_COLUMNS,
)

from ..models import (
    VARTrainer,
    LSTMTrainer,
    TransformerTrainer,
    VARIMATrainer,
    WaveNetTrainer,
    GaussianProcessTrainer,
    PredictiveAutoencoderTrainer,
)
from ..visualization.visualization import ModelVisualizer
from ..evaluation.model_comparison import ModelComparison
from ..data.time_based_splitting import TimeBasedDataSplitter


class ComprehensiveAnalyzer:
    """
    Comprehensive analyzer for all time series models with complete evaluation.
    Coordinates all model training, visualization, and comparison tasks.
    Uses time-based data splitting (8min train, 2min test).
    """

    def __init__(
        self,
        data_dir: str = "processed_data",
        results_dir: str = "results/comprehensive_results",
    ):
        self.data_dir = data_dir
        self.results_dir = results_dir

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(f"{results_dir}/plots", exist_ok=True)

        self.data_splitter = TimeBasedDataSplitter(data_dir)
        self.evaluator = self._initialize_evaluator()
        self.visualizer = ModelVisualizer(results_dir)
        self.comparator = ModelComparison(results_dir)

        print("Loading and preparing data...")
        self.train_data, self.test_data = self.data_splitter.get_train_test_data()
        print(
            f"Data loaded: {len(self.train_data):,} train, {len(self.test_data):,} test samples"
        )

    def _initialize_evaluator(self):
        return StandardizedModelEvaluator(
            split_method="time_based",
            data_dir=self.data_dir,
            results_dir=self.results_dir,
        )

    # ----------------------------------------------------------------------
    # Helpers for metrics and diagnostics
    # ----------------------------------------------------------------------

    @staticmethod
    def _to_np(x):
        return np.asarray(x).reshape(-1)

    @staticmethod
    def _rmse(y, yhat):
        y, yhat = ComprehensiveAnalyzer._to_np(y), ComprehensiveAnalyzer._to_np(yhat)
        m = np.isfinite(y) & np.isfinite(yhat)
        return float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2))) if m.sum() else float("nan")

    @staticmethod
    def _mae(y, yhat):
        y, yhat = ComprehensiveAnalyzer._to_np(y), ComprehensiveAnalyzer._to_np(yhat)
        m = np.isfinite(y) & np.isfinite(yhat)
        return float(np.mean(np.abs(y[m] - yhat[m]))) if m.sum() else float("nan")

    @staticmethod
    def _persistence_baseline(y):
        y = ComprehensiveAnalyzer._to_np(y)
        if y.size == 0:
            return y
        yhat = np.empty_like(y)
        yhat[0] = y[0]
        yhat[1:] = y[:-1]
        return yhat

    @staticmethod
    def _pearson_r_lag1(residuals):
        r = ComprehensiveAnalyzer._to_np(residuals)
        if r.size < 2:
            return float("nan"), float("nan")
        r_t, r_tm1 = r[1:], r[:-1]
        m = np.isfinite(r_t) & np.isfinite(r_tm1)
        if m.sum() < 2:
            return float("nan"), float("nan")
        corr, pval = pearsonr(r_t[m], r_tm1[m])
        return float(corr), float(pval)

    @staticmethod
    def _ljung_box_pvalue(residuals, lags: int = 10):
        if not _HAS_SM:
            return float("nan")
        r = ComprehensiveAnalyzer._to_np(residuals)
        if np.sum(np.isfinite(r)) < lags + 3:
            return float("nan")
        try:
            lb = sm.stats.acorr_ljungbox(r, lags=[lags], return_df=True)
            return float(lb["lb_pvalue"].iloc[0])
        except Exception:
            return float("nan")

    def _compute_diagnostics(self, y_true_dict, y_pred_dict):
        per_target, rmse_list, mae_list, rrmse_list, rmae_list, pearson_list, lb_list = (
            {},
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for t in STANDARD_TARGET_COLUMNS:
            if t not in y_true_dict or t not in y_pred_dict:
                continue
            y, yhat = self._to_np(y_true_dict[t]), self._to_np(y_pred_dict[t])

            rmse = self._rmse(y, yhat)
            mae = self._mae(y, yhat)

            yhat_naive = self._persistence_baseline(y)
            naive_rmse, naive_mae = self._rmse(y, yhat_naive), self._mae(y, yhat_naive)

            rrmse = rmse / (naive_rmse + 1e-12) if np.isfinite(naive_rmse) else float("nan")
            rmae = mae / (naive_mae + 1e-12) if np.isfinite(naive_mae) else float("nan")

            residuals = y - yhat
            r1, r1p = self._pearson_r_lag1(residuals)
            lbp = self._ljung_box_pvalue(residuals, lags=10)

            per_target[t] = dict(
                rmse=rmse,
                mae=mae,
                relative_rmse=rrmse,
                relative_mae=rmae,
                resid_pearson_lag1=r1,
                resid_pearson_lag1_p=r1p,
                resid_ljung_p_lag10=lbp,
            )

            for lst, val in [
                (rmse_list, rmse),
                (mae_list, mae),
                (rrmse_list, rrmse),
                (rmae_list, rmae),
                (pearson_list, r1),
                (lb_list, lbp),
            ]:
                if np.isfinite(val):
                    lst.append(val)

        overall = dict(
            mean_rmse=float(np.mean(rmse_list)) if rmse_list else float("nan"),
            mean_mae=float(np.mean(mae_list)) if mae_list else float("nan"),
            mean_relative_rmse=float(np.mean(rrmse_list)) if rrmse_list else float("nan"),
            mean_relative_mae=float(np.mean(rmae_list)) if rmae_list else float("nan"),
            mean_resid_pearson_lag1=float(np.mean(pearson_list)) if pearson_list else float("nan"),
            mean_resid_ljung_p_lag10=float(np.mean(lb_list)) if lb_list else float("nan"),
        )

        return {"per_target": per_target, "overall": overall}

    def _attach_and_save_diagnostics(self, model_name, results, y_true_dict, y_pred_dict):
        diagnostics = self._compute_diagnostics(y_true_dict, y_pred_dict)
        results["diagnostics"] = diagnostics
        out = {
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "summary": results.get("summary", {}),
            "diagnostics": diagnostics,
        }
        out_path = os.path.join(self.results_dir, f"{model_name}_diagnostics.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved diagnostics → {out_path}")

    # ----------------------------------------------------------------------
    # Wrappers for each trainer
    # ----------------------------------------------------------------------

    def train_and_evaluate_var(self, max_lags=10):
        results, y_true, y_pred = VARTrainer.train_and_evaluate(
            self.train_data, self.test_data, self.evaluator, self.results_dir, max_lags
        )
        if results and y_true and y_pred:
            self._attach_and_save_diagnostics("VAR", results, y_true, y_pred)
            self.visualizer.create_var_plots(y_true, y_pred, results)
        return results

    def train_and_evaluate_simple_lstm(self):
        out = LSTMTrainer.train_and_evaluate(
            self.train_data, self.test_data, self.evaluator, self.results_dir
        )
        if isinstance(out, tuple) and len(out) >= 3:
            results, y_true, y_pred = out[:3]
            train_losses, val_losses = out[3], out[4]
        else:
            return {}
        if results and y_true and y_pred:
            self._attach_and_save_diagnostics("LSTM", results, y_true, y_pred)
            self.visualizer.create_lstm_plots(y_true, y_pred, results, train_losses, val_losses)
        return results

    def train_and_evaluate_baseline_transformer(self):
        results, y_true, y_pred = TransformerTrainer.train_and_evaluate(
            self.train_data, self.test_data, self.evaluator, self.results_dir
        )
        if results and y_true and y_pred:
            self._attach_and_save_diagnostics("Transformer", results, y_true, y_pred)
            self.visualizer.create_transformer_plots(y_true, y_pred, results)
        return results

    def train_and_evaluate_varima(self):
        results, y_true, y_pred = VARIMATrainer.train_and_evaluate(
            self.train_data, self.test_data, self.evaluator, self.results_dir
        )
        if results and y_true and y_pred:
            self._attach_and_save_diagnostics("VARIMA", results, y_true, y_pred)
            self.visualizer.create_varima_plots(y_true, y_pred, results)
        return results

    def train_and_evaluate_wavenet(self):
        results, y_true, y_pred = WaveNetTrainer.train_and_evaluate(
            self.train_data, self.test_data, self.evaluator, self.results_dir
        )
        if results and y_true and y_pred:
            self._attach_and_save_diagnostics("WaveNet", results, y_true, y_pred)
            self.visualizer.create_wavenet_plots(y_true, y_pred, results)
        return results

    def train_and_evaluate_gaussian_process(self):
        results, y_true, y_pred = GaussianProcessTrainer.train_and_evaluate(
            self.train_data, self.test_data, self.evaluator, self.results_dir
        )
        if results and y_true and y_pred:
            self._attach_and_save_diagnostics("GaussianProcess", results, y_true, y_pred)
            self.visualizer.create_gp_plots(y_true, y_pred, results)
        return results

    def train_and_evaluate_predictive_autoencoder_deterministic(self):
        results, y_true, y_pred = PredictiveAutoencoderTrainer.train_and_evaluate(
            self.train_data, self.test_data, self.evaluator, self.results_dir, model_type="deterministic"
        )
        if results and y_true and y_pred:
            self._attach_and_save_diagnostics("PredictiveAutoencoder_Deterministic", results, y_true, y_pred)
            self.visualizer.create_autoencoder_plots(y_true, y_pred, results, "deterministic")
        return results

    def train_and_evaluate_predictive_autoencoder_sparse(self):
        results, y_true, y_pred = PredictiveAutoencoderTrainer.train_and_evaluate(
            self.train_data, self.test_data, self.evaluator, self.results_dir, model_type="sparse"
        )
        if results and y_true and y_pred:
            self._attach_and_save_diagnostics("PredictiveAutoencoder_Sparse", results, y_true, y_pred)
            self.visualizer.create_autoencoder_plots(y_true, y_pred, results, "sparse")
        return results

    def train_and_evaluate_predictive_autoencoder_variational(self):
        results, y_true, y_pred = PredictiveAutoencoderTrainer.train_and_evaluate(
            self.train_data, self.test_data, self.evaluator, self.results_dir, model_type="variational"
        )
        if results and y_true and y_pred:
            self._attach_and_save_diagnostics("PredictiveAutoencoder_Variational", results, y_true, y_pred)
            self.visualizer.create_autoencoder_plots(y_true, y_pred, results, "variational")
        return results

    # ----------------------------------------------------------------------
    # Orchestrator
    # ----------------------------------------------------------------------

    def run_extended_analysis(self):
        print("\n" + "=" * 80)
        print("EXTENDED COMPREHENSIVE ANALYSIS - ALL MODELS")
        print("=" * 80)

        all_results, successful, failed = {}, [], []

        model_functions = [
            ("VAR", self.train_and_evaluate_var),
            ("VARIMA", self.train_and_evaluate_varima),
            ("LSTM", self.train_and_evaluate_simple_lstm),
            ("Transformer", self.train_and_evaluate_baseline_transformer),
            ("WaveNet", self.train_and_evaluate_wavenet),
            ("GaussianProcess", self.train_and_evaluate_gaussian_process),
            ("PredictiveAutoencoder_Deterministic", self.train_and_evaluate_predictive_autoencoder_deterministic),
            ("PredictiveAutoencoder_Sparse", self.train_and_evaluate_predictive_autoencoder_sparse),
            ("PredictiveAutoencoder_Variational", self.train_and_evaluate_predictive_autoencoder_variational),
        ]

        for model_name, fn in model_functions:
            print(f"\n{'='*20} ANALYZING {model_name.upper()} {'='*20}")
            try:
                results = fn()
                if results and "summary" in results and results["summary"].get("avg_r2") is not None:
                    all_results[model_name] = results
                    successful.append(model_name)
                    print(f"SUCCESS {model_name} | Avg R²={results['summary']['avg_r2']:.4f}")
                    if "diagnostics" in results:
                        di = results["diagnostics"]["overall"]
                        print(f"   Mean Rel.RMSE={di['mean_relative_rmse']:.3f}, Rel.MAE={di['mean_relative_mae']:.3f}")
                else:
                    failed.append(model_name)
            except Exception as e:
                failed.append(model_name)
                import traceback; traceback.print_exc()

        print(f"\nSuccessful models: {successful}")
        print(f"Failed models: {failed}")

        if len(successful) >= 3:
            comparison_results = self.comparator.create_extended_comparison_report(all_results)
            extended_results = {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(model_functions),
                "successful_models": successful,
                "failed_models": failed,
                "detailed_results": all_results,
                "comparison": comparison_results,
            }
            results_file = os.path.join(self.results_dir, "extended_comprehensive_results.json")
            with open(results_file, "w") as f:
                json.dump(extended_results, f, indent=2)
            print(f"Extended results saved to {results_file}")
            return extended_results
        else:
            print("Insufficient successful models for comparison")
            return None