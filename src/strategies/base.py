"""
Base Strategy Interface for Batch Quality Audits.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from src.types import Batch, AuditResult, QualityThresholds
from src.decision_policy import AcceptanceDecisionPolicy


class BaseAuditStrategy(ABC):
    """Abstract base class for all sampling, estimation, and decision strategies."""

    def __init__(
        self,
        name: str,
        budget_fraction: float = 0.01,  # 1% standard audit budget
        thresholds: QualityThresholds = QualityThresholds(),
        policy: Optional[AcceptanceDecisionPolicy] = None,
        human_auditor_error_rate: float = 0.0,  # Simulate auditor mistakes (Label Noise)
    ):
        self.name = name
        self.budget_fraction = budget_fraction
        self.thresholds = thresholds
        self.policy = policy or AcceptanceDecisionPolicy(thresholds=thresholds)
        self.human_auditor_error_rate = human_auditor_error_rate

    def _observe_defects(self, batch: Batch, sample_indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Retrieves ground truth defects for the sampled indices and optionally
        injects human auditor label noise (flipping the truth with probability = human_auditor_error_rate).
        """
        true_defects = batch.defects[sample_indices]
        if self.human_auditor_error_rate <= 0.0:
            return true_defects
            
        # Inject symmetric label noise
        flips = (rng.random(len(sample_indices)) < self.human_auditor_error_rate)
        return np.where(flips, ~true_defects, true_defects)

    @abstractmethod
    def audit(self, batch: Batch, **kwargs) -> AuditResult:
        """
        Execute sampling, estimation, and decision policy on the batch.
        
        Args:
            batch: Batch of items to audit.
            
        Returns:
            AuditResult containing sample indices, estimated error rate, CI, and decision.
        """
        pass
