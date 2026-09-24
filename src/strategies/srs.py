"""
Baseline Strategy: Simple Random Sampling (SRS) + Frequentist MLE + Wilson Score CI.
"""

from typing import List, Optional
import numpy as np
from scipy import stats
from src.types import Batch, AuditResult, Decision, QualityThresholds
from src.strategies.base import BaseAuditStrategy
from src.decision_policy import AcceptanceDecisionPolicy


def compute_wilson_ci(
    k: int,
    n: int,
    population_size: Optional[int] = None,
    confidence: float = 0.95
) -> tuple[float, float]:
    """
    Wilson Score Confidence Interval with optional Finite Population Correction (FPC).
    """
    if n == 0:
        return 0.0, 1.0
    
    alpha = 1.0 - confidence
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p_hat = k / n
    
    fpc = 1.0
    if population_size is not None and population_size > n:
        fpc = np.sqrt((population_size - n) / (population_size - 1.0))
    
    denominator = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2.0 * n)) / denominator
    spread = (z * fpc / denominator) * np.sqrt((p_hat * (1.0 - p_hat) / n) + ((z ** 2) / (4.0 * (n ** 2))))
    
    ci_lower = max(0.0, center - spread)
    ci_upper = min(1.0, center + spread)
    return float(ci_lower), float(ci_upper)


class SimpleRandomSamplingStrategy(BaseAuditStrategy):
    """
    Mandatory Baseline:
    1. Sample exactly n = budget * N uniformly at random.
    2. Estimate p_hat = k / n.
    3. Compute 95% Wilson Score CI with FPC.
    4. Apply pre-fixed threshold decision rule.
    """

    def __init__(
        self,
        budget_fraction: float = 0.01,
        seed: Optional[int] = 42,
        thresholds: QualityThresholds = QualityThresholds(),
        policy: Optional[AcceptanceDecisionPolicy] = None,
    ):
        super().__init__(
            name="Baseline: Simple Random Sampling (SRS)",
            budget_fraction=budget_fraction,
            thresholds=thresholds,
            policy=policy,
        )
        self.rng = np.random.default_rng(seed)

    def audit(self, batch: Batch, **kwargs) -> AuditResult:
        N = batch.size
        n = max(1, int(np.floor(self.budget_fraction * N)))
        
        # Fast array indexing
        sample_indices = self.rng.choice(N, size=n, replace=False)
        defects = int(np.sum(batch.defects[sample_indices]))
        
        p_hat = float(defects / n)
        ci_lower, ci_upper = compute_wilson_ci(defects, n, population_size=N, confidence=0.95)
        decision = self.policy.decide_from_point_estimate(p_hat)
        
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
                "defects_observed": defects,
                "frequentist_se": float(np.sqrt(p_hat * (1 - p_hat) / n)),
            }
        )
