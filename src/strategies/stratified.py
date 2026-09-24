"""
Risk-Stratified Sampling Strategy with Neyman Allocation.
Partitions the batch using metadata (annotator history & item complexity)
to minimize estimation variance.
"""

from typing import List, Dict, Optional
import numpy as np
from src.types import Batch, AuditResult, Decision, QualityThresholds
from src.strategies.base import BaseAuditStrategy
from src.decision_policy import AcceptanceDecisionPolicy


class RiskStratifiedStrategy(BaseAuditStrategy):
    """
    Risk-Stratified Sampling:
    1. Stratify batch items into H risk strata based on available metadata.
    2. Allocate sample budget across strata using Neyman Allocation.
    3. Compute provably unbiased Stratified Estimator.
    """

    def __init__(
        self,
        budget_fraction: float = 0.01,
        num_strata: int = 3,
        seed: Optional[int] = 42,
        thresholds: QualityThresholds = QualityThresholds(),
        policy: Optional[AcceptanceDecisionPolicy] = None,
    ):
        super().__init__(
            name="Stratified: Risk & Neyman Allocation",
            budget_fraction=budget_fraction,
            thresholds=thresholds,
            policy=policy,
        )
        self.num_strata = num_strata
        self.rng = np.random.default_rng(seed)

    def audit(self, batch: Batch, **kwargs) -> AuditResult:
        N = batch.size
        n_total = max(self.num_strata * 2, int(np.floor(self.budget_fraction * N)))
        
        # 1. Vectorized item risk calculation
        item_risks = 0.6 * batch.complexity_scores + 0.4 * batch.surrogate_scores
        
        # 2. Stratify via quantiles
        strata_cuts = np.quantile(item_risks, np.linspace(0, 1, self.num_strata + 1))
        strata_cuts[0] -= 1e-6
        strata_cuts[-1] += 1e-6
        
        strata_assignments = np.digitize(item_risks, strata_cuts[1:-1])
        
        # 3. Neyman Allocation weights
        expected_p_strata = [0.01, 0.035, 0.08][:self.num_strata]
        sigma_strata = [np.sqrt(p * (1 - p)) for p in expected_p_strata]
        
        strata_indices = [np.where(strata_assignments == h)[0] for h in range(self.num_strata)]
        weights = [len(strata_indices[h]) * sigma_strata[h] for h in range(self.num_strata)]
        
        total_weight = sum(weights) + 1e-12
        raw_alloc = [int(np.round(n_total * (w / total_weight))) for w in weights]
        alloc = [max(2, a) for a in raw_alloc]
        
        while sum(alloc) > n_total:
            max_idx = int(np.argmax(alloc))
            alloc[max_idx] -= 1
        while sum(alloc) < n_total:
            min_idx = int(np.argmin(alloc))
            alloc[min_idx] += 1
            
        # 4. Fast vectorized sampling per stratum
        sampled_list = []
        strata_results = []
        
        for h in range(self.num_strata):
            nh = alloc[h]
            Nh = len(strata_indices[h])
            if nh > Nh:
                nh = Nh
            sampled_h = self.rng.choice(strata_indices[h], size=nh, replace=False)
            sampled_list.append(sampled_h)
            
            kh = int(np.sum(batch.defects[sampled_h]))
            y_bar_h = kh / nh if nh > 0 else 0.0
            s2_h = (y_bar_h * (1.0 - y_bar_h) * nh / (nh - 1)) if nh > 1 else 0.0
            
            strata_results.append({
                "Nh": Nh,
                "nh": nh,
                "y_bar_h": y_bar_h,
                "s2_h": s2_h,
            })
            
        all_sampled_indices = np.concatenate(sampled_list)
        
        # 5. Stratified Estimator
        p_hat = sum((res["Nh"] / N) * res["y_bar_h"] for res in strata_results)
        
        var_strat = sum(
            ((res["Nh"] / N) ** 2) * (1.0 - res["nh"] / res["Nh"]) * (res["s2_h"] / res["nh"])
            for res in strata_results
            if res["nh"] > 1
        )
        se_strat = float(np.sqrt(max(1e-9, var_strat)))
        
        z = 1.96
        ci_lower = max(0.0, float(p_hat - z * se_strat))
        ci_upper = min(1.0, float(p_hat + z * se_strat))
        
        decision = self.policy.decide_from_confidence_interval(
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            point_estimate=p_hat
        )
        
        return AuditResult(
            strategy_name=self.name,
            batch_id=batch.batch_id,
            sample_size=len(all_sampled_indices),
            sample_indices=all_sampled_indices,
            estimated_error_rate=p_hat,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            decision=decision,
            items_audited=len(all_sampled_indices),
            budget_fraction_used=float(len(all_sampled_indices) / N),
            metadata={
                "strata_allocation": alloc,
                "stratified_se": se_strat,
            }
        )
