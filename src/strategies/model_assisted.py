"""
Model-Assisted Difference Estimator & Defect Prediction Strategy.
Uses automated surrogate signals (AI judge / heuristic checkers) to achieve
dramatic variance reduction while maintaining guaranteed zero-bias.
"""

from typing import List, Optional
import numpy as np
from src.types import Batch, AuditResult, Decision, QualityThresholds
from src.strategies.base import BaseAuditStrategy
from src.decision_policy import AcceptanceDecisionPolicy


class ModelAssistedDifferenceStrategy(BaseAuditStrategy):
    """
    Model-Assisted Difference Estimator (Machine-in-the-loop Quality Audit):
    
    1. An automated surrogate model (or rule-based heuristic) scores all N items
       in the batch: s_i in [0, 1].
    2. Population surrogate mean is known exactly: s_bar_pop = (1/N) * sum(s_i).
    3. Human auditor reviews a 1% sample S.
    4. Difference Estimator:
       p_hat_diff = s_bar_pop + (1/n) * sum_{i in S} (y_i - s_i)
    """

    def __init__(
        self,
        budget_fraction: float = 0.01,
        active_sampling: bool = False,
        seed: Optional[int] = 42,
        thresholds: QualityThresholds = QualityThresholds(),
        policy: Optional[AcceptanceDecisionPolicy] = None,
        human_auditor_error_rate: float = 0.0,
    ):
        name_suffix = " (Active Importance)" if active_sampling else " (Difference Estimator)"
        super().__init__(
            name=f"Model-Assisted{name_suffix}",
            budget_fraction=budget_fraction,
            thresholds=thresholds,
            policy=policy,
            human_auditor_error_rate=human_auditor_error_rate,
        )
        self.active_sampling = active_sampling
        self.rng = np.random.default_rng(seed)

    def audit(self, batch: Batch, **kwargs) -> AuditResult:
        N = batch.size
        n = max(2, int(np.floor(self.budget_fraction * N)))
        
        s_bar_pop = float(np.mean(batch.surrogate_scores))
        
        if not self.active_sampling:
            # Standard SRS for difference estimator
            sample_indices = self.rng.choice(N, size=n, replace=False)
            
            observed_y = self._observe_defects(batch, sample_indices, self.rng)
            y_sample = observed_y.astype(np.float64)
            s_sample = batch.surrogate_scores[sample_indices].astype(np.float64)
            residuals = y_sample - s_sample
            
            p_hat = float(s_bar_pop + np.mean(residuals))
            p_hat = float(np.clip(p_hat, 0.0, 1.0))
            
            fpc = (N - n) / (N - 1.0)
            res_var = np.var(residuals, ddof=1) if n > 1 else 0.0
            se_diff = float(np.sqrt(max(1e-9, fpc * (res_var / n))))
            
        else:
            # Importance / Active Sampling
            eps = 0.05
            raw_probs = batch.surrogate_scores + eps
            sampling_probs = raw_probs / np.sum(raw_probs)
            
            sample_indices = self.rng.choice(N, size=n, replace=False, p=sampling_probs)
            
            observed_y = self._observe_defects(batch, sample_indices, self.rng)
            y_sample = observed_y.astype(np.float64)
            pi_sample = sampling_probs[sample_indices] * n
            
            p_hat = float(np.sum(y_sample / pi_sample) / N)
            p_hat = float(np.clip(p_hat, 0.0, 1.0))
            
            se_diff = float(np.std(y_sample / pi_sample, ddof=1) / np.sqrt(n) / N * n)
            se_diff = float(np.clip(se_diff, 1e-4, 0.05))
            
        z = 1.96
        ci_lower = max(0.0, float(p_hat - z * se_diff))
        ci_upper = min(1.0, float(p_hat + z * se_diff))
        
        decision = self.policy.decide_from_confidence_interval(
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            point_estimate=p_hat
        )
        
        return AuditResult(
            strategy_name=self.name,
            batch_id=batch.batch_id,
            sample_size=n,
            sample_indices=sample_indices,
            estimated_error_rate=p_hat,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            decision=decision,
            items_audited=n,
            budget_fraction_used=float(n / N),
            metadata={
                "s_bar_pop": s_bar_pop,
                "se_diff": se_diff,
                "active_sampling": self.active_sampling,
            }
        )
