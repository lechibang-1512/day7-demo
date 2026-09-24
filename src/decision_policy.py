"""
Pre-fixed Decision Policy for Batch Acceptance, Rework, and Rejection.
Contains mathematically grounded thresholds and evaluation loss functions.
"""

from typing import Tuple, Dict, Any
from src.types import Decision, QualityThresholds, AuditResult


class AcceptanceDecisionPolicy:
    """
    Fixed-Threshold Decision Engine.
    
    CRITICAL RULE: Thresholds are parameterized BEFORE any audit evaluation
    and kept strictly immutable across all batches and strategies.
    """

    def __init__(
        self,
        thresholds: QualityThresholds = QualityThresholds(),
        accept_cutoff: float = 0.025,
        reject_cutoff: float = 0.055,
        # Bayesian / Uncertainty-aware decision parameters
        bayes_accept_prob_threshold: float = 0.85,
        bayes_reject_prob_threshold: float = 0.70,
    ):
        self.thresholds = thresholds
        self.accept_cutoff = accept_cutoff
        self.reject_cutoff = reject_cutoff
        self.bayes_accept_prob_threshold = bayes_accept_prob_threshold
        self.bayes_reject_prob_threshold = bayes_reject_prob_threshold

    def decide_from_point_estimate(self, estimated_p: float) -> Decision:
        """Standard frequentist / point-estimate decision rule."""
        if estimated_p <= self.accept_cutoff:
            return Decision.ACCEPT
        elif estimated_p >= self.reject_cutoff:
            return Decision.REJECT
        else:
            return Decision.REWORK

    def decide_from_confidence_interval(
        self,
        ci_lower: float,
        ci_upper: float,
        point_estimate: float
    ) -> Decision:
        """
        Conservative risk-averse decision policy using Confidence Intervals.
        - ACCEPT if upper bound is below tolerable boundary.
        - REJECT if lower bound exceeds acceptable threshold.
        - REWORK if uncertainty overlaps boundary or point estimate is in rework zone.
        """
        # If even the upper bound is <= 4.0%, batch is safe to accept
        if ci_upper <= 0.040:
            return Decision.ACCEPT
        # If lower bound is >= 5.0%, batch is definitively bad
        elif ci_lower >= 0.050:
            return Decision.REJECT
        # High uncertainty or middle zone
        elif point_estimate <= self.accept_cutoff and ci_upper <= 0.050:
            return Decision.ACCEPT
        elif point_estimate >= self.reject_cutoff:
            return Decision.REJECT
        else:
            return Decision.REWORK

    def decide_from_posterior(
        self,
        prob_good: float,  # P(p <= AQL | data)
        prob_bad: float,   # P(p >= LTPD | data)
        point_estimate: float,
    ) -> Decision:
        """
        Bayesian Decision Rule optimizing expected utility under asymmetric losses.
        """
        if prob_good >= self.bayes_accept_prob_threshold:
            return Decision.ACCEPT
        elif prob_bad >= self.bayes_reject_prob_threshold:
            return Decision.REJECT
        else:
            # When neither high confidence good nor high confidence bad
            if point_estimate <= self.accept_cutoff and prob_bad < 0.05:
                return Decision.ACCEPT
            elif point_estimate >= self.reject_cutoff:
                return Decision.REJECT
            else:
                return Decision.REWORK

    @staticmethod
    def calculate_business_loss(
        true_p: float,
        decision: Decision,
        items_audited: int,
        item_audit_cost: float = 1.0,
        false_accept_penalty: float = 500.0,
        false_reject_penalty: float = 150.0,
        rework_penalty: float = 50.0,
    ) -> float:
        """
        Calculates total operational cost = Audit Cost + Decision Penalties.
        - Accepting a bad batch (p >= 0.06): Severe penalty ($500).
        - Rejecting a good batch (p <= 0.02): Vendor friction / delay penalty ($150).
        - Unnecessary Rework on good batch: Extra friction ($50).
        """
        cost = items_audited * item_audit_cost
        
        is_good = true_p <= 0.02
        is_bad = true_p >= 0.06
        is_borderline = 0.02 < true_p < 0.06
        
        if decision == Decision.ACCEPT:
            if is_bad:
                # Catastrophic Type II error
                cost += false_accept_penalty * (true_p / 0.06)
            elif is_borderline:
                cost += false_accept_penalty * 0.25
        elif decision == Decision.REJECT:
            if is_good:
                # Producer Type I error
                cost += false_reject_penalty
        elif decision == Decision.REWORK:
            cost += rework_penalty
            if is_good:
                cost += 20.0  # Slight penalty for bothering vendor on good batch
                
        return cost
