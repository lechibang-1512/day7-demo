"""
Interactive CLI Demo for Vendor Dataset Acceptance (Challenge C7).
Demonstrates step-by-step audit of a single batch with ground truth verification.
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.types import QualityThresholds, Decision
from src.data_generator import BatchGenerator, DEFAULT_VENDORS
from src.decision_policy import AcceptanceDecisionPolicy
from src.strategies import (
    SimpleRandomSamplingStrategy,
    TwoStageAcceptanceStrategy,
    EmpiricalBayesStrategy,
    RiskStratifiedStrategy,
    ModelAssistedDifferenceStrategy,
)


def run_demo():
    console = Console()
    console.print(Panel.fit(
        "[bold cyan]CHALLENGE C7: VENDOR DATASET ACCEPTANCE DEMO[/bold cyan]\n"
        "[italic white]Audit 1% Budget -> Point/Interval Estimation -> Pre-fixed Decision[/italic white]",
        border_style="cyan"
    ))

    # 1. Setup
    thresholds = QualityThresholds(aql=0.02, ltpd=0.06)
    policy = AcceptanceDecisionPolicy(thresholds=thresholds)
    generator = BatchGenerator(seed=2026)

    # 2. Simulate Vendor Batch Arrival
    vendor = DEFAULT_VENDORS["vendor_flaky"]
    batch_size = 10_000
    batch = generator.generate_batch(
        batch_id="BATCH-VINAI-2026-09",
        vendor=vendor,
        batch_size=batch_size,
        surrogate_correlation=0.75,
    )

    console.print(f"[bold yellow]1. INCOMING DATASET BATCH DELIVERED[/bold yellow]")
    console.print(f" • Batch ID: [bold]{batch.batch_id}[/bold]")
    console.print(f" • Vendor: [bold]{vendor.name}[/bold] ({vendor.vendor_id})")
    console.print(f" • Batch Size (N): [bold]{batch_size:,}[/bold] items")
    console.print(f" • Audit Budget: [bold]1.0%[/bold] (max 100 human reviews)")
    console.print(f" • [dim]Ground truth status is HIDDEN from audit strategies.[/dim]\n")

    # 3. Audit using multiple strategies
    strategies = [
        SimpleRandomSamplingStrategy(budget_fraction=0.01, thresholds=thresholds, policy=policy),
        TwoStageAcceptanceStrategy(budget_fraction=0.01, thresholds=thresholds, policy=policy),
        EmpiricalBayesStrategy(budget_fraction=0.01, thresholds=thresholds, policy=policy),
        RiskStratifiedStrategy(budget_fraction=0.01, thresholds=thresholds, policy=policy),
        ModelAssistedDifferenceStrategy(budget_fraction=0.01, thresholds=thresholds, policy=policy),
    ]

    table = Table(title="[bold]Audit Results & Batch Decisions (1% Budget)[/bold]")
    table.add_column("Audit Strategy", style="cyan", no_wrap=False)
    table.add_column("Audited (n)", justify="right", style="yellow")
    table.add_column("Est. Error (p̂)", justify="right", style="magenta")
    table.add_column("95% Confidence / Credible Int.", justify="center", style="blue")
    table.add_column("Policy Decision", justify="center", style="bold")
    table.add_column("Key Metadata / Mechanics", style="dim white")

    results = []
    for strat in strategies:
        res = strat.audit(batch)
        results.append(res)
        
        # Format decision color
        if res.decision == Decision.ACCEPT:
            dec_str = "[bold green]ACCEPT[/bold green]"
        elif res.decision == Decision.REJECT:
            dec_str = "[bold red]REJECT[/bold red]"
        else:
            dec_str = "[bold yellow]REWORK[/bold yellow]"
            
        meta_str = ", ".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in list(res.metadata.items())[:2])
        
        table.add_row(
            res.strategy_name,
            f"{res.items_audited} ({res.budget_fraction_used*100:.1f}%)",
            f"{res.estimated_error_rate*100:.2f}%",
            f"[{res.ci_lower*100:.2f}%, {res.ci_upper*100:.2f}%]",
            dec_str,
            meta_str,
        )

    console.print(table)
    console.print("\n")

    # 4. Reveal Ground Truth for Evaluation
    p_true = batch.true_error_rate
    true_decision = batch.true_decision
    defects_count = int(round(p_true * batch_size))

    console.print(Panel(
        f"[bold green]4. GROUND TRUTH REVEALED (For Scoring & Benchmark Validation)[/bold green]\n"
        f" • True Error Rate (p): [bold]{p_true*100:.2f}%[/bold] ({defects_count:,} defects out of {batch_size:,})\n"
        f" • True Batch Status: [bold]{true_decision.value}[/bold] "
        f"({'p <= 2% (AQL)' if p_true <= 0.02 else 'p >= 6% (LTPD)' if p_true >= 0.06 else '2% < p < 6% (Rework Zone)'})\n"
        f" • Pre-fixed Quality Thresholds: AQL = {thresholds.aql*100:.1f}%, LTPD = {thresholds.ltpd*100:.1f}%\n",
        border_style="green"
    ))

    # Error analysis table
    eval_table = Table(title="Estimation Accuracy & Cost on this Batch")
    eval_table.add_column("Strategy", style="cyan")
    eval_table.add_column("|p̂ - p_true| (MAE)", justify="right", style="magenta")
    eval_table.add_column("Inside 95% CI?", justify="center")
    eval_table.add_column("Decision Match?", justify="center")
    eval_table.add_column("Review Cost", justify="right", style="yellow")

    for res in results:
        mae = abs(res.estimated_error_rate - p_true) * 100
        in_ci = "✓ Yes" if (res.ci_lower <= p_true <= res.ci_upper) else "✗ No"
        match = "✓ Match" if res.decision == true_decision else f"✗ ({res.decision.value} vs {true_decision.value})"
        eval_table.add_row(
            res.strategy_name,
            f"{mae:.3f}%",
            f"[green]{in_ci}[/green]" if "Yes" in in_ci else f"[red]{in_ci}[/red]",
            f"[green]{match}[/green]" if "Match" in match else f"[yellow]{match}[/yellow]",
            f"{res.items_audited} samples",
        )

    console.print(eval_table)


if __name__ == "__main__":
    run_demo()
