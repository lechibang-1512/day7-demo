"""
Data types and dataclasses for Vendor Dataset Acceptance (Challenge C7).
Optimized for high-performance vectorized simulation and audits.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    REWORK = "REWORK"
    REJECT = "REJECT"


@dataclass
class QualityThresholds:
    """
    PRE-FIXED Quality thresholds for dataset acceptance.
    CRITICAL: Must NEVER be modified after observing ground truth.
    
    AQL (Acceptable Quality Limit):
        Max error rate where a batch should almost certainly be ACCEPTED.
        p <= aql (e.g. 2%) -> ACCEPT.
    
    LTPD / RQL (Lot Tolerance Percent Defective / Rejectable Quality Level):
        Min error rate where a batch is definitively unacceptable and must be REJECTED.
        p >= ltpd (e.g. 6%) -> REJECT.
        
    Indifference / Rework Zone:
        aql < p < ltpd (e.g. 2% to 6%) -> REWORK.
    """
    aql: float = 0.02      # 2% error rate target
    ltpd: float = 0.06     # 6% defect cutoff
    alpha_risk: float = 0.05  # Max allowable producer risk (False reject rate on good batch)
    beta_risk: float = 0.10   # Max allowable consumer risk (False accept rate on bad batch)


@dataclass
class Batch:
    """A batch of dataset annotations delivered by a vendor (vectorized)."""
    batch_id: str
    vendor_id: str
    size: int
    defects: np.ndarray             # Ground truth defect indicator bool array (Hidden during audit)
    complexity_scores: np.ndarray   # Item complexity [0, 1] float32
    surrogate_scores: np.ndarray    # AI judge / heuristic predicted defect prob [0, 1] float32
    annotator_ids: np.ndarray       # Annotator IDs int16
    
    @property
    def true_error_rate(self) -> float:
        """Ground truth error rate (Held-out)."""
        return float(np.mean(self.defects))
    
    @property
    def true_decision(self) -> Decision:
        """Ideal ground truth decision given pre-fixed thresholds."""
        p = self.true_error_rate
        if p <= 0.02:
            return Decision.ACCEPT
        elif p >= 0.06:
            return Decision.REJECT
        else:
            return Decision.REWORK


@dataclass
class AuditResult:
    """Output of an audit strategy on a batch."""
    strategy_name: str
    batch_id: str
    sample_size: int
    sample_indices: np.ndarray
    estimated_error_rate: float
    ci_lower: float
    ci_upper: float
    decision: Decision
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Audit cost tracking
    items_audited: int = 0
    budget_fraction_used: float = 0.0


@dataclass
class ScenarioMetrics:
    """Aggregated evaluation metrics for a strategy across multiple batches."""
    strategy_name: str
    scenario_name: str
    num_batches: int
    
    # Primary Metrics
    mean_abs_error: float          # MAE = E[|p_hat - p|]
    median_abs_error: float        # Median AE
    rmse: float                    # Root Mean Squared Error
    bad_batch_recall: float        # Recall on batches where true p >= LTPD
    good_batch_accept_rate: float  # Acceptance rate on batches where true p <= AQL
    rework_precision: float        # Precision in identifying rework batches
    
    # Secondary & Risk Metrics
    false_reject_rate: float       # Producer's risk: P(REJECT | p <= AQL)
    false_accept_rate: float       # Consumer's risk: P(ACCEPT | p >= LTPD)
    ci_coverage: float             # Fraction of batches where true p in [ci_lower, ci_upper]
    average_sample_number: float   # Average number of items audited per batch (Cost)
    avg_budget_fraction: float     # Average % of batch audited
    
    # Composite Score: (1 - MAE) * 0.5 + Bad_Batch_Recall * 0.5
    composite_score: float = 0.0
