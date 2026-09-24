"""
Visualization and Diagnostic Plotting for Vendor Dataset Acceptance.
Generates:
1. Operating Characteristic (OC) Curves: P(Accept) vs True Error Rate
2. MAE vs Budget Trade-off Curves
3. Strategy Comparison Radar/Bar Charts
"""

import os
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from src.types import QualityThresholds, Decision
from src.data_generator import BatchGenerator
from src.strategies import (
    SimpleRandomSamplingStrategy,
    TwoStageAcceptanceStrategy,
    EmpiricalBayesStrategy,
    RiskStratifiedStrategy,
    ModelAssistedDifferenceStrategy,
)


def plot_operating_characteristic_curves(
    output_path: str = "oc_curves.png",
    num_simulations_per_point: int = 300,
):
    """
    Plots the Operating Characteristic (OC) Curve:
    P(ACCEPT) as a function of the true batch defect rate p.
    Shows Type I (Producer's) and Type II (Consumer's) risk profiles.
    """
    error_rates = np.linspace(0.005, 0.10, 20)
    strategies = [
        SimpleRandomSamplingStrategy(budget_fraction=0.01),
        TwoStageAcceptanceStrategy(budget_fraction=0.01),
        EmpiricalBayesStrategy(budget_fraction=0.01),
        RiskStratifiedStrategy(budget_fraction=0.01),
        ModelAssistedDifferenceStrategy(budget_fraction=0.01),
    ]
    
    generator = BatchGenerator(seed=123)
    vendor = list(generator.generate_random_vendors(num_vendors=1).values())[0]
    results = {strat.name: [] for strat in strategies}
    
    for p in error_rates:
        for strat in strategies:
            accept_count = 0
            for i in range(num_simulations_per_point):
                batch = generator.generate_batch(
                    f"oc_{p}_{i}",
                    vendor=vendor,
                    batch_size=10_000,
                    forced_error_rate=p,
                )
                res = strat.audit(batch)
                if res.decision == Decision.ACCEPT:
                    accept_count += 1
            results[strat.name].append(accept_count / num_simulations_per_point)
            
    plt.figure(figsize=(10, 6), dpi=150)
    for name, p_accepts in results.items():
        plt.plot(error_rates * 100, p_accepts, marker="o", label=name, linewidth=2)
        
    plt.axvline(x=2.0, color="green", linestyle="--", alpha=0.7, label="AQL (2.0% Accept Target)")
    plt.axvline(x=6.0, color="red", linestyle="--", alpha=0.7, label="LTPD (6.0% Reject Cutoff)")
    plt.axhspan(0.90, 1.0, color="green", alpha=0.08, label="Acceptance Zone (>90%)")
    plt.axhspan(0.0, 0.10, color="red", alpha=0.08, label="Rejection Zone (<10%)")
    
    plt.title("Operating Characteristic (OC) Curves across Audit Strategies (1% Budget)", fontsize=13, fontweight="bold")
    plt.xlabel("True Batch Defect Rate (%)", fontsize=11)
    plt.ylabel("Probability of Batch Acceptance P(ACCEPT)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved OC curve plot to {output_path}")


def plot_budget_vs_mae(
    output_path: str = "budget_vs_mae.png",
    num_batches: int = 150,
):
    """
    Plots Error of Quality Estimation (MAE) as audit budget scales from 0.2% to 2.5%.
    """
    budgets = [0.002, 0.005, 0.010, 0.015, 0.020, 0.025]
    generator = BatchGenerator(seed=456)
    vendor = list(generator.generate_random_vendors(num_vendors=1).values())[0]
    test_batches = [
        generator.generate_batch(f"b_{i}", vendor, batch_size=10_000)
        for i in range(num_batches)
    ]
    
    strategy_classes = [
        ("Baseline (SRS)", SimpleRandomSamplingStrategy),
        ("Empirical Bayes", EmpiricalBayesStrategy),
        ("Risk-Stratified", RiskStratifiedStrategy),
        ("Model-Assisted (Diff)", ModelAssistedDifferenceStrategy),
    ]
    
    results = {name: [] for name, _ in strategy_classes}
    
    for b in budgets:
        for name, cls in strategy_classes:
            strat = cls(budget_fraction=b)
            maes = []
            for batch in test_batches:
                res = strat.audit(batch)
                maes.append(abs(res.estimated_error_rate - batch.true_error_rate))
            results[name].append(np.mean(maes) * 100)
            
    plt.figure(figsize=(10, 6), dpi=150)
    for name, mae_vals in results.items():
        plt.plot([b * 100 for b in budgets], mae_vals, marker="s", label=name, linewidth=2)
        
    plt.axvline(x=1.0, color="gray", linestyle=":", label="Production Budget (1.0%)")
    plt.title("Estimation Error (MAE) vs Audit Budget Fraction", fontsize=13, fontweight="bold")
    plt.xlabel("Audit Budget (%) of 10,000 items", fontsize=11)
    plt.ylabel("Mean Absolute Error MAE (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved budget vs MAE plot to {output_path}")


if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
    plot_operating_characteristic_curves("artifacts/oc_curves.png")
    plot_budget_vs_mae("artifacts/budget_vs_mae.png")
