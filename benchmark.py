"""
Main Benchmark Runner for Vendor Dataset Acceptance (Challenge C7).
Runs 5+ audit strategies across 1,000+ simulated batches across 5 operational scenarios.
"""

import os
import time
import pandas as pd
from tabulate import tabulate
from rich.console import Console
from rich.table import Table

from src.types import QualityThresholds
from src.data_generator import BatchGenerator
from src.decision_policy import AcceptanceDecisionPolicy
from src.strategies import (
    SimpleRandomSamplingStrategy,
    TwoStageAcceptanceStrategy,
    EmpiricalBayesStrategy,
    RiskStratifiedStrategy,
    ModelAssistedDifferenceStrategy,
)
from src.evaluator import BenchmarkEvaluator


def run_comprehensive_benchmark():
    console = Console()
    console.print("[bold cyan]================================================================[/bold cyan]")
    console.print("[bold cyan]  CHALLENGE C7: VENDOR DATASET ACCEPTANCE BENCHMARK ENGINE      [/bold cyan]")
    console.print("[bold cyan]  1% Budget Quality Estimation & Decision Policy Evaluation     [/bold cyan]")
    console.print("[bold cyan]================================================================[/bold cyan]\n")

    # 1. Setup Pre-fixed Thresholds & Decision Policy
    thresholds = QualityThresholds(aql=0.02, ltpd=0.06)
    policy = AcceptanceDecisionPolicy(thresholds=thresholds)
    evaluator = BenchmarkEvaluator(thresholds=thresholds, policy=policy)

    # 2. Instantiate Strategies under 1% Budget
    BUDGET = 0.01
    strategies = [
        SimpleRandomSamplingStrategy(budget_fraction=BUDGET, thresholds=thresholds, policy=policy),
        TwoStageAcceptanceStrategy(budget_fraction=BUDGET, thresholds=thresholds, policy=policy),
        EmpiricalBayesStrategy(budget_fraction=BUDGET, thresholds=thresholds, policy=policy),
        RiskStratifiedStrategy(budget_fraction=BUDGET, thresholds=thresholds, policy=policy),
        ModelAssistedDifferenceStrategy(budget_fraction=BUDGET, active_sampling=False, thresholds=thresholds, policy=policy),
        ModelAssistedDifferenceStrategy(budget_fraction=BUDGET, active_sampling=True, thresholds=thresholds, policy=policy),
    ]

    # 3. Generate Benchmark Suite (1,000 batches total)
    console.print("[yellow]Generating synthetic batches across 5 operational scenarios (N=10,000 each)...[/yellow]")
    generator = BatchGenerator(seed=42)
    suite = generator.generate_benchmark_suite(num_batches_per_scenario=200, batch_size=10_000)
    console.print(f"[green]✓ Generated {sum(len(b) for b in suite.values())} batches across {len(suite)} scenarios.[/green]\n")

    # 4. Execute Benchmark
    start_time = time.time()
    console.print("[yellow]Running Monte Carlo audit simulations...[/yellow]")
    df_results = evaluator.run_full_benchmark(strategies, suite)
    elapsed = time.time() - start_time
    console.print(f"[green]✓ Benchmark completed in {elapsed:.2f} seconds.[/green]\n")

    # 5. Output Per-Scenario Tables
    os.makedirs("artifacts", exist_ok=True)
    df_results.to_csv("artifacts/benchmark_detailed.csv", index=False)

    for scenario_name in suite.keys():
        scenario_df = df_results[df_results["Scenario"] == scenario_name].drop(columns=["Scenario"])
        
        table = Table(title=f"Scenario: {scenario_name.upper()} (200 Batches, N=10,000)")
        table.add_column("Strategy", style="cyan", no_wrap=False)
        table.add_column("MAE (%)", justify="right", style="magenta")
        table.add_column("Bad-Batch Recall", justify="right", style="red")
        table.add_column("Good Accept Rate", justify="right", style="green")
        table.add_column("False Reject", justify="right")
        table.add_column("95% CI Cov.", justify="right")
        table.add_column("ASN (Cost)", justify="right", style="yellow")
        table.add_column("Comp. Score", justify="right", style="bold white")

        for _, row in scenario_df.iterrows():
            table.add_row(
                row["Strategy"],
                f"{row['MAE (%)']:.3f}%",
                f"{row['Bad-Batch Recall (%)']:.1f}%",
                f"{row['Good-Batch Accept (%)']:.1f}%",
                f"{row['False Reject Rate (%)']:.1f}%",
                f"{row['95% CI Coverage (%)']:.1f}%",
                f"{row['Avg Items Audited (ASN)']:.1f}",
                f"{row['Composite Score']:.4f}",
            )
        console.print(table)
        console.print("\n")

    # 6. Overall Aggregated Summary Leaderboard
    summary_df = df_results.groupby("Strategy").agg({
        "MAE (%)": "mean",
        "Median AE (%)": "mean",
        "RMSE (%)": "mean",
        "Bad-Batch Recall (%)": "mean",
        "Good-Batch Accept (%)": "mean",
        "False Reject Rate (%)": "mean",
        "95% CI Coverage (%)": "mean",
        "Avg Items Audited (ASN)": "mean",
        "Composite Score": "mean",
    }).reset_index().sort_values(by="Composite Score", ascending=False)

    summary_df.to_csv("artifacts/benchmark_summary.csv", index=False)

    console.print("[bold green]================================================================[/bold green]")
    console.print("[bold green]          OVERALL BENCHMARK LEADERBOARD (ACROSS ALL 1,000 BATCHES) [/bold green]")
    console.print("[bold green]================================================================[/bold green]")
    
    summary_table = Table(title="Overall Strategy Ranking (Higher Composite Score is Better)")
    summary_table.add_column("Rank", justify="center", style="bold")
    summary_table.add_column("Strategy", style="bold cyan")
    summary_table.add_column("Mean MAE (%)", justify="right", style="magenta")
    summary_table.add_column("Mean Bad Recall", justify="right", style="red")
    summary_table.add_column("Mean Good Accept", justify="right", style="green")
    summary_table.add_column("95% CI Cov.", justify="right")
    summary_table.add_column("Mean ASN (Cost)", justify="right", style="yellow")
    summary_table.add_column("Composite Score", justify="right", style="bold white on blue")

    for rank, (_, row) in enumerate(summary_df.iterrows(), start=1):
        summary_table.add_row(
            str(rank),
            row["Strategy"],
            f"{row['MAE (%)']:.3f}%",
            f"{row['Bad-Batch Recall (%)']:.1f}%",
            f"{row['Good-Batch Accept (%)']:.1f}%",
            f"{row['95% CI Coverage (%)']:.1f}%",
            f"{row['Avg Items Audited (ASN)']:.1f}",
            f"{row['Composite Score']:.4f}",
        )
    console.print(summary_table)

    return df_results, summary_df


if __name__ == "__main__":
    run_comprehensive_benchmark()
