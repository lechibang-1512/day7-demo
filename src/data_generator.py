"""
Batch and Dataset Generator for Vendor Quality Auditing.
Simulates realistic annotation batches with annotator heterogeneity,
task complexity, and surrogate model signals.
Optimized for high-performance vectorized generation.
"""

from typing import List, Dict, Optional
import numpy as np
from scipy.special import expit
from src.types import Batch, QualityThresholds


class VendorProfile:
    """Defines the statistical quality behavior of a specific vendor."""
    def __init__(
        self,
        vendor_id: str,
        name: str,
        base_error_rate: float,
        error_variance: float,
        annotator_skill_variance: float = 0.5,
        complexity_sensitivity: float = 1.2,
    ):
        self.vendor_id = vendor_id
        self.name = name
        self.base_error_rate = base_error_rate
        self.error_variance = error_variance
        self.annotator_skill_variance = annotator_skill_variance
        self.complexity_sensitivity = complexity_sensitivity


DEFAULT_VENDORS = {
    "vendor_tier1": VendorProfile(
        vendor_id="vendor_tier1",
        name="Tier-1 High Quality Vendor",
        base_error_rate=0.012,  # 1.2% base error (Target: ACCEPT)
        error_variance=0.004,
        annotator_skill_variance=0.3,
    ),
    "vendor_flaky": VendorProfile(
        vendor_id="vendor_flaky",
        name="Flaky / Bimodal Vendor",
        base_error_rate=0.045,  # Mixed batches (2% to 12%)
        error_variance=0.035,
        annotator_skill_variance=0.8,
    ),
    "vendor_tier3": VendorProfile(
        vendor_id="vendor_tier3",
        name="Tier-3 Low Quality Vendor",
        base_error_rate=0.085,  # 8.5% base error (Target: REJECT)
        error_variance=0.020,
        annotator_skill_variance=0.6,
    ),
    "vendor_borderline": VendorProfile(
        vendor_id="vendor_borderline",
        name="Borderline / Drift Vendor",
        base_error_rate=0.038,  # 3.8% error (Target: REWORK)
        error_variance=0.008,
        annotator_skill_variance=0.4,
    ),
}


class BatchGenerator:
    """Generates synthetic annotation batches with realistic distributions."""

    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)

    def generate_batch(
        self,
        batch_id: str,
        vendor: VendorProfile,
        batch_size: int = 10_000,
        num_annotators: int = 20,
        surrogate_correlation: float = 0.70,
        forced_error_rate: Optional[float] = None,
    ) -> Batch:
        """
        Vectorized generation of a single batch.
        """
        # 1. Determine batch target error rate
        if forced_error_rate is not None:
            target_p = forced_error_rate
        else:
            target_p = float(np.clip(
                self.rng.normal(vendor.base_error_rate, vendor.error_variance),
                0.001,
                0.30
            ))

        # 2. Annotator skill levels (random effects)
        annotator_skills = self.rng.normal(0, vendor.annotator_skill_variance, num_annotators)
        annotator_assignments = self.rng.integers(0, num_annotators, size=batch_size, dtype=np.int16)
        
        # 3. Item covariates
        complexity_scores = self.rng.beta(2, 5, size=batch_size).astype(np.float32)
        
        # 4. Latent logit calculation
        base_logit = np.log(target_p / (1.0 - target_p + 1e-12))
        item_logits = (
            base_logit
            + vendor.complexity_sensitivity * (complexity_scores - np.mean(complexity_scores))
            + annotator_skills[annotator_assignments]
        )
        
        item_probs = expit(item_logits)
        scaling = target_p / (np.mean(item_probs) + 1e-12)
        calibrated_probs = np.clip(item_probs * scaling, 0.0001, 0.9999)
        
        # 5. Generate ground-truth defect indicators
        defects = (self.rng.binomial(1, calibrated_probs) == 1)
        
        # 6. Generate surrogate score s_i
        noise = self.rng.normal(0, 1.0 - surrogate_correlation, size=batch_size)
        surrogate_logits = base_logit + surrogate_correlation * (item_logits - base_logit) + noise
        surrogate_scores = expit(surrogate_logits).astype(np.float32)
        
        return Batch(
            batch_id=batch_id,
            vendor_id=vendor.vendor_id,
            size=batch_size,
            defects=defects,
            complexity_scores=complexity_scores,
            surrogate_scores=surrogate_scores,
            annotator_ids=annotator_assignments,
        )

    def generate_benchmark_suite(
        self,
        num_batches_per_scenario: int = 200,
        batch_size: int = 10_000,
    ) -> Dict[str, List[Batch]]:
        suite: Dict[str, List[Batch]] = {}
        
        suite["high_quality"] = [
            self.generate_batch(f"hq_{i}", DEFAULT_VENDORS["vendor_tier1"], batch_size=batch_size)
            for i in range(num_batches_per_scenario)
        ]
        
        suite["bad_quality"] = [
            self.generate_batch(f"bad_{i}", DEFAULT_VENDORS["vendor_tier3"], batch_size=batch_size)
            for i in range(num_batches_per_scenario)
        ]
        
        suite["borderline_rework"] = [
            self.generate_batch(f"border_{i}", DEFAULT_VENDORS["vendor_borderline"], batch_size=batch_size)
            for i in range(num_batches_per_scenario)
        ]
        
        suite["mixed_flaky"] = [
            self.generate_batch(f"flaky_{i}", DEFAULT_VENDORS["vendor_flaky"], batch_size=batch_size)
            for i in range(num_batches_per_scenario)
        ]
        
        clustered_vendor = VendorProfile(
            vendor_id="vendor_clustered",
            name="Clustered Outlier Vendor",
            base_error_rate=0.035,
            error_variance=0.02,
            annotator_skill_variance=1.8,
        )
        suite["stress_clustered"] = [
            self.generate_batch(f"clust_{i}", clustered_vendor, batch_size=batch_size)
            for i in range(num_batches_per_scenario)
        ]
        
        return suite
