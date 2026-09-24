"""
Benchmark and Evaluation Engine for Vendor Dataset Acceptance.
Executes batch simulations and aggregates statistical performance metrics.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from src.types import Batch, Decision, AuditResult, ScenarioMetrics, QualityThresholds
from src.strategies.base import BaseAuditStrategy
from src.decision_policy import AcceptanceDecisionPolicy


class BenchmarkEvaluator:
    """
    Evaluates multiple audit strategies against hidden ground truth
    across diverse operational scenarios.
    """

    def __init__(
        self,
        thresholds: QualityThresholds = QualityThresholds(),
        policy: AcceptanceDecisionPolicy = AcceptanceDecisionPolicy(),
    ):
        self.thresholds = thresholds
        self.policy = policy

    def evaluate_strategy_on_batches(
        self,
        strategy: BaseAuditStrategy,
        batches: List[Batch],
        scenario_name: str,
    ) -> ScenarioMetrics:
        """
        Run audit strategy over a collection of batches and compute all metrics.
        """
        abs_errors = []
        squared_errors = []
        in_ci_list = []
        audited_counts = []
        audited_counts_accepted = []
        audited_counts_rejected = []
        budget_fractions = []
        
        # Classification counters
        bad_batches_total = 0
        bad_batches_caught = 0  # Decision in {REJECT, REWORK}
        bad_batches_rejected = 0 # Decision == REJECT
        
        good_batches_total = 0
        good_batches_accepted = 0 # Decision == ACCEPT
        good_batches_rejected = 0 # Decision == REJECT (False reject)
        
        rework_batches_total = 0
        rework_correctly_flagged = 0
        
        for batch in batches:
            p_true = batch.true_error_rate
            result = strategy.audit(batch)
            
            p_hat = result.estimated_error_rate
            abs_err = abs(p_hat - p_true)
            sq_err = (p_hat - p_true) ** 2
            
            abs_errors.append(abs_err)
            squared_errors.append(sq_err)
            
            in_ci = (result.ci_lower <= p_true <= result.ci_upper)
            in_ci_list.append(in_ci)
            
            audited_counts.append(result.items_audited)
            budget_fractions.append(result.budget_fraction_used)
            
            if result.decision == Decision.ACCEPT:
                audited_counts_accepted.append(result.items_audited)
            elif result.decision == Decision.REJECT:
                audited_counts_rejected.append(result.items_audited)
            
            # Ground truth classification based on pre-fixed AQL / LTPD
            is_good = (p_true <= self.thresholds.aql)
            is_bad = (p_true >= self.thresholds.ltpd)
            is_borderline = (self.thresholds.aql < p_true < self.thresholds.ltpd)
            
            if is_bad:
                bad_batches_total += 1
                if result.decision in (Decision.REJECT, Decision.REWORK):
                    bad_batches_caught += 1
                if result.decision == Decision.REJECT:
                    bad_batches_rejected += 1
                    
            elif is_good:
                good_batches_total += 1
                if result.decision == Decision.ACCEPT:
                    good_batches_accepted += 1
                elif result.decision == Decision.REJECT:
                    good_batches_rejected += 1
                    
            elif is_borderline:
                rework_batches_total += 1
                if result.decision == Decision.REWORK:
                    rework_correctly_flagged += 1

        num_batches = len(batches)
        mae = float(np.mean(abs_errors))
        med_ae = float(np.median(abs_errors))
        rmse = float(np.sqrt(np.mean(squared_errors)))
        ci_coverage = float(np.mean(in_ci_list))
        asn = float(np.mean(audited_counts))
        avg_budget = float(np.mean(budget_fractions))
        
        bad_batch_recall = (bad_batches_caught / bad_batches_total) if bad_batches_total > 0 else 1.0
        good_batch_accept_rate = (good_batches_accepted / good_batches_total) if good_batches_total > 0 else 1.0
        false_reject_rate = (good_batches_rejected / good_batches_total) if good_batches_total > 0 else 0.0
        false_accept_rate = 1.0 - bad_batch_recall
        rework_precision = (rework_correctly_flagged / rework_batches_total) if rework_batches_total > 0 else 1.0
        
        cost_per_accepted = float(np.mean(audited_counts_accepted)) if audited_counts_accepted else 0.0
        cost_per_rejected = float(np.mean(audited_counts_rejected)) if audited_counts_rejected else 0.0
        
        # Composite score according to Challenge C7 primary metric:
        # Lower MAE is better, Higher Recall is better.
        # Score = (1.0 - MAE) * 0.5 + (bad_batch_recall) * 0.5
        composite_score = float((1.0 - mae) * 0.5 + bad_batch_recall * 0.5)

        return ScenarioMetrics(
            strategy_name=strategy.name,
            scenario_name=scenario_name,
            num_batches=num_batches,
            mean_abs_error=mae,
            median_abs_error=med_ae,
            rmse=rmse,
            bad_batch_recall=bad_batch_recall,
            good_batch_accept_rate=good_batch_accept_rate,
            rework_precision=rework_precision,
            false_reject_rate=false_reject_rate,
            false_accept_rate=false_accept_rate,
            ci_coverage=ci_coverage,
            average_sample_number=asn,
            avg_budget_fraction=avg_budget,
            cost_per_accepted_batch=cost_per_accepted,
            cost_per_rejected_batch=cost_per_rejected,
            composite_score=composite_score,
        )

    def run_full_benchmark(
        self,
        strategies: List[BaseAuditStrategy],
        benchmark_suite: Dict[str, List[Batch]],
    ) -> pd.DataFrame:
        """
        Runs all strategies across all benchmark scenarios and returns summary DataFrame.
        """
        records = []
        for scenario_name, batches in benchmark_suite.items():
            for strategy in strategies:
                metrics = self.evaluate_strategy_on_batches(strategy, batches, scenario_name)
                records.append({
                    "Scenario": scenario_name,
                    "Strategy": strategy.name,
                    "MAE (%)": round(metrics.mean_abs_error * 100, 3),
                    "Median AE (%)": round(metrics.median_abs_error * 100, 3),
                    "RMSE (%)": round(metrics.rmse * 100, 3),
                    "Bad-Batch Recall (%)": round(metrics.bad_batch_recall * 100, 1),
                    "Good-Batch Accept (%)": round(metrics.good_batch_accept_rate * 100, 1),
                    "False Reject Rate (%)": round(metrics.false_reject_rate * 100, 1),
                    "95% CI Coverage (%)": round(metrics.ci_coverage * 100, 1),
                    "Avg Items Audited (ASN)": round(metrics.average_sample_number, 1),
                    "Avg Budget (%)": round(metrics.avg_budget_fraction * 100, 2),
                    "Cost / Accept": round(metrics.cost_per_accepted_batch, 1),
                    "Cost / Reject": round(metrics.cost_per_rejected_batch, 1),
                    "Composite Score": round(metrics.composite_score, 4),
                })
                
        return pd.DataFrame(records)
