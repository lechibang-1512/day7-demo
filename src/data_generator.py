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




class BatchGenerator:
    """Generates synthetic annotation batches with realistic distributions."""

    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)

    def generate_random_vendors(self, num_vendors: int = 5) -> Dict[str, VendorProfile]:
        vendors = {}
        for i in range(num_vendors):
            # Base error rate mostly between 0.5% and 15%
            base_error = float(self.rng.beta(1.5, 15.0)) * 0.2
            base_error = max(0.005, min(0.20, base_error))
            
            err_var = float(self.rng.uniform(0.002, 0.040))
            skill_var = float(self.rng.uniform(0.2, 1.2))
            
            v_id = f"vendor_rand_{i}"
            vendors[v_id] = VendorProfile(
                vendor_id=v_id,
                name=f"Random Vendor {i} (Err: {base_error:.1%})",
                base_error_rate=base_error,
                error_variance=err_var,
                annotator_skill_variance=skill_var,
            )
        return vendors

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
        num_vendors: int = 10,
    ) -> Dict[str, List[Batch]]:
        suite: Dict[str, List[Batch]] = {}
        
        # Dynamically generate a pool of randomized vendors
        pool = list(self.generate_random_vendors(num_vendors=num_vendors).values())
        
        # 1. High Quality Scenario: batches where error rate <= AQL (0.5% - 2.0%)
        suite["high_quality"] = []
        for i in range(num_batches_per_scenario):
            v = self.rng.choice(pool)
            target_p = float(self.rng.uniform(0.005, 0.020))
            suite["high_quality"].append(
                self.generate_batch(f"hq_{i}", v, batch_size=batch_size, forced_error_rate=target_p)
            )
            
        # 2. Bad Quality Scenario: batches where error rate >= LTPD (6.0% - 15.0%)
        suite["bad_quality"] = []
        for i in range(num_batches_per_scenario):
            v = self.rng.choice(pool)
            target_p = float(self.rng.uniform(0.060, 0.150))
            suite["bad_quality"].append(
                self.generate_batch(f"bad_{i}", v, batch_size=batch_size, forced_error_rate=target_p)
            )
            
        # 3. Borderline / Rework Scenario: batches near threshold (2.1% - 5.9%)
        suite["borderline_rework"] = []
        for i in range(num_batches_per_scenario):
            v = self.rng.choice(pool)
            target_p = float(self.rng.uniform(0.022, 0.058))
            suite["borderline_rework"].append(
                self.generate_batch(f"border_{i}", v, batch_size=batch_size, forced_error_rate=target_p)
            )
            
        # 4. Mixed / Flaky Scenario: bimodal distribution across entire quality range
        suite["mixed_flaky"] = []
        for i in range(num_batches_per_scenario):
            v = self.rng.choice(pool)
            r = self.rng.random()
            if r < 0.50:
                p = float(self.rng.uniform(0.005, 0.020))
            elif r < 0.80:
                p = float(self.rng.uniform(0.060, 0.120))
            else:
                p = float(self.rng.uniform(0.025, 0.055))
            suite["mixed_flaky"].append(
                self.generate_batch(f"flaky_{i}", v, batch_size=batch_size, forced_error_rate=p)
            )
            
        # 5. Stress Clustered Scenario: extreme annotator skill variance (clustering effect)
        suite["stress_clustered"] = []
        clustered_vendor = VendorProfile(
            vendor_id="vendor_clustered_rand",
            name="Dynamic Clustered Outlier Vendor",
            base_error_rate=0.035,
            error_variance=0.02,
            annotator_skill_variance=1.8,
        )
        for i in range(num_batches_per_scenario):
            suite["stress_clustered"].append(
                self.generate_batch(f"clust_{i}", clustered_vendor, batch_size=batch_size)
            )
            
        return suite
