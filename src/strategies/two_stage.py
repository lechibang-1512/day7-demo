"""
Two-Stage Sequential Acceptance Sampling Strategy.
Implements early stopping (MIL-STD-105E / ANSI Z1.4 adaptation) to save audit budget.
"""

from typing import List, Optional
import numpy as np
from src.types import Batch, AuditResult, Decision, QualityThresholds
from src.strategies.base import BaseAuditStrategy
from src.strategies.srs import compute_wilson_ci
from src.decision_policy import AcceptanceDecisionPolicy


class TwoStageAcceptanceStrategy(BaseAuditStrategy):
    """
    Two-Stage Double Sampling Strategy:
    - Stage 1: Inspect 50% of the budget (e.g. n1 = 0.5% * N).
      - If k1 <= c_acc1 (e.g. 0): Early ACCEPT -> Terminate (Saves 50% budget).
      - If k1 >= c_rej1 (e.g. >= 4): Early REJECT -> Terminate (Saves 50% budget).
      - Otherwise: Indeterminate zone -> Proceed to Stage 2.
    - Stage 2: Inspect remaining 50% of budget (n2 = 0.5% * N).
    """

    def __init__(
        self,
        budget_fraction: float = 0.01,
        stage1_fraction: float = 0.005,
        c_acc1: int = 0,
        c_rej1: int = 4,
        seed: Optional[int] = 42,
        thresholds: QualityThresholds = QualityThresholds(),
        policy: Optional[AcceptanceDecisionPolicy] = None,
        human_auditor_error_rate: float = 0.0,
    ):
        super().__init__(
            name="Sequential: Two-Stage Double Sampling",
            budget_fraction=budget_fraction,
            thresholds=thresholds,
            policy=policy,
            human_auditor_error_rate=human_auditor_error_rate,
        )
        self.stage1_fraction = stage1_fraction
        self.c_acc1 = c_acc1
        self.c_rej1 = c_rej1
        self.rng = np.random.default_rng(seed)

    def audit(self, batch: Batch, **kwargs) -> AuditResult:
        N = batch.size
        n_total = max(2, int(np.floor(self.budget_fraction * N)))
        n1 = max(1, int(np.floor(self.stage1_fraction * N)))
        n2 = n_total - n1
        
        all_shuffled_indices = self.rng.permutation(N)
        stage1_indices = all_shuffled_indices[:n1]
        
        # 1. Audit Stage 1
        observed_stage1 = self._observe_defects(batch, stage1_indices, self.rng)
        k1 = int(np.sum(observed_stage1))
        
        # Check Stage 1 early stopping
        if k1 <= self.c_acc1:
            p_hat = float(k1 / n1)
            ci_lower, ci_upper = compute_wilson_ci(k1, n1, population_size=N)
            return AuditResult(
                strategy_name=self.name,
                batch_id=batch.batch_id,
                sample_size=n1,
                sample_indices=stage1_indices,
                estimated_error_rate=p_hat,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                decision=Decision.ACCEPT,
                items_audited=n1,
                budget_fraction_used=float(n1 / N),
                metadata={"stage_terminated": 1, "k1": k1, "reason": "Early Accept"}
            )
            
        elif k1 >= self.c_rej1:
            p_hat = float(k1 / n1)
            ci_lower, ci_upper = compute_wilson_ci(k1, n1, population_size=N)
            return AuditResult(
                strategy_name=self.name,
                batch_id=batch.batch_id,
                sample_size=n1,
                sample_indices=stage1_indices,
                estimated_error_rate=p_hat,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                decision=Decision.REJECT,
                items_audited=n1,
                budget_fraction_used=float(n1 / N),
                metadata={"stage_terminated": 1, "k1": k1, "reason": "Early Reject"}
            )
            
        # 2. Ambiguous result -> Stage 2
        stage2_indices = all_shuffled_indices[n1:n1 + n2]
        observed_stage2 = self._observe_defects(batch, stage2_indices, self.rng)
        k2 = int(np.sum(observed_stage2))
        
        total_defects = k1 + k2
        combined_indices = all_shuffled_indices[:n_total]
        p_hat = float(total_defects / n_total)
        ci_lower, ci_upper = compute_wilson_ci(total_defects, n_total, population_size=N)
        
        decision = self.policy.decide_from_point_estimate(p_hat)
        
        return AuditResult(
            strategy_name=self.name,
            batch_id=batch.batch_id,
            sample_size=n_total,
            sample_indices=combined_indices,
            estimated_error_rate=p_hat,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            decision=decision,
            items_audited=n_total,
            budget_fraction_used=float(n_total / N),
            metadata={
                "stage_terminated": 2,
                "k1": k1,
                "k2": k2,
                "total_defects": total_defects,
                "reason": "Two-Stage Complete"
            }
        )
