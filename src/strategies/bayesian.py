"""
Empirical Bayes Beta-Binomial Strategy with Vendor Historical Priors.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy import stats
from src.types import Batch, AuditResult, Decision, QualityThresholds
from src.strategies.base import BaseAuditStrategy
from src.decision_policy import AcceptanceDecisionPolicy


class EmpiricalBayesStrategy(BaseAuditStrategy):
    """
    Bayesian Beta-Binomial Updating:
    - Leverages empirical prior distribution for each vendor: p ~ Beta(alpha_0, beta_0).
    - Conjugate posterior: p | k ~ Beta(alpha_0 + k, beta_0 + n - k).
    - Optimal shrinkage estimator: balances sample evidence against historical track record.
    - Computes exact posterior tail probabilities: P(p <= AQL | data) and P(p >= LTPD | data).
    """

    def __init__(
        self,
        budget_fraction: float = 0.01,
        vendor_priors: Optional[Dict[str, Tuple[float, float]]] = None,
        default_prior: Tuple[float, float] = (1.5, 40.0),
        seed: Optional[int] = 42,
        thresholds: QualityThresholds = QualityThresholds(),
        policy: Optional[AcceptanceDecisionPolicy] = None,
    ):
        super().__init__(
            name="Bayesian: Empirical Beta-Binomial",
            budget_fraction=budget_fraction,
            thresholds=thresholds,
            policy=policy,
        )
        self.vendor_priors = vendor_priors or {
            "vendor_tier1": (1.2, 98.8),    # Mean ~ 1.2%
            "vendor_flaky": (2.0, 40.0),     # Mean ~ 4.7%
            "vendor_tier3": (8.5, 91.5),    # Mean ~ 8.5%
            "vendor_borderline": (3.8, 96.2),# Mean ~ 3.8%
            "vendor_clustered": (2.5, 60.0), # Mean ~ 4.0%
        }
        self.default_prior = default_prior
        self.rng = np.random.default_rng(seed)

    def audit(self, batch: Batch, **kwargs) -> AuditResult:
        N = batch.size
        n = max(1, int(np.floor(self.budget_fraction * N)))
        
        sample_indices = self.rng.choice(N, size=n, replace=False)
        k = int(np.sum(batch.defects[sample_indices]))
        
        alpha_0, beta_0 = self.vendor_priors.get(batch.vendor_id, self.default_prior)
        
        alpha_post = alpha_0 + k
        beta_post = beta_0 + (n - k)
        
        p_bayes = float(alpha_post / (alpha_post + beta_post))
        ci_lower = float(stats.beta.ppf(0.025, alpha_post, beta_post))
        ci_upper = float(stats.beta.ppf(0.975, alpha_post, beta_post))
        
        prob_good = float(stats.beta.cdf(self.thresholds.aql, alpha_post, beta_post))
        prob_bad = float(1.0 - stats.beta.cdf(self.thresholds.ltpd, alpha_post, beta_post))
        
        decision = self.policy.decide_from_posterior(
            prob_good=prob_good,
            prob_bad=prob_bad,
            point_estimate=p_bayes
        )
        
        return AuditResult(
            strategy_name=self.name,
            batch_id=batch.batch_id,
            sample_size=n,
            sample_indices=sample_indices,
            estimated_error_rate=p_bayes,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            decision=decision,
            items_audited=n,
            budget_fraction_used=float(n / N),
            metadata={
                "prior_alpha": alpha_0,
                "prior_beta": beta_0,
                "prob_good_aql": prob_good,
                "prob_bad_ltpd": prob_bad,
                "sample_k": k,
            }
        )
