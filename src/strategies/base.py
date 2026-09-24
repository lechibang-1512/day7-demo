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
    ):
        self.name = name
        self.budget_fraction = budget_fraction
        self.thresholds = thresholds
        self.policy = policy or AcceptanceDecisionPolicy(thresholds=thresholds)

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
