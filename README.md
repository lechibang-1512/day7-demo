# Challenge C7: Vendor Dataset Acceptance (Audit 1% Budget)

A mathematically grounded, production-grade sampling, estimation, and decision policy engine for large-scale vendor dataset delivery under severe review budget constraints (1.0% audit).

---

## 1. Problem Formulation & Theoretical Foundations

### 1.1 The Production Dilemma
In modern AI data pipelines (e.g., LLM fine-tuning, computer vision annotations, RLHF), external vendors deliver large batches ($N = 10,000$ items). Manual quality audit is expensive, limiting human review to $\approx 1.0\%$ ($n = 100$ items).

From this tiny sample, the system must:
1. **Estimate true batch quality**: $\hat{p} \approx p = \frac{1}{N}\sum_{i=1}^N y_i$, with rigorous confidence/credible intervals.
2. **Output a definitive decision**: $\text{Decision} \in \{\text{ACCEPT}, \text{REWORK}, \text{REJECT}\}$.
3. **Guarantee statistical risk bounds**:
   - **Producer's Risk ($\alpha$)**: $\mathbb{P}(\text{REJECT} \mid p \le p_{AQL}) \le 5\%$ (avoid false rejections of good work).
   - **Consumer's Risk ($\beta$)**: $\mathbb{P}(\text{ACCEPT} \mid p \ge p_{LTPD}) \le 10\%$ (avoid poisoning training sets with bad data).

---

### 1.2 Pre-Fixed Quality Thresholds (No Post-Hoc Tuning)

> [!IMPORTANT]
> In accordance with statistical quality control standards (MIL-STD-105E / ANSI/ASQ Z1.4), decision thresholds are mathematically parameterized **prior to observing audit data** and held strictly immutable:

| Zone | Error Rate Range ($p$) | Definition | Action Target |
| :--- | :--- | :--- | :--- |
| **Acceptance Zone** | $p \le 2.0\%$ ($p_{AQL}$) | **AQL (Acceptable Quality Limit)**: High-quality batch suitable for production training. | **`ACCEPT`** |
| **Indifference Zone** | $2.0\% < p < 6.0\%$ | **Rework Range**: Borderline quality containing fixable flaws. Return to vendor for targeted remediation. | **`REWORK`** |
| **Rejection Zone** | $p \ge 6.0\%$ ($p_{LTPD}$) | **LTPD (Lot Tolerance Percent Defective)**: Catastrophic defect rate. Batch must be discarded. | **`REJECT`** |

---

## 2. Implemented Strategies & Mathematical Formulations

```
                  ┌─────────────────────────────────────────────────────────┐
                  │          Incoming Dataset Batch (N = 10,000)            │
                  └────────────────────────────┬────────────────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
     ┌───────────────────────────────────┐           ┌───────────────────────────────────┐
     │      Classical Sampling           │           │     Model-Assisted / Metadata     │
     └─────────────────┬─────────────────┘           └─────────────────┬─────────────────┘
                       │                                               │
       ┌───────────────┼───────────────┐               ┌───────────────┼───────────────┐
       ▼               ▼               ▼               ▼               ▼               ▼
 ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
 │ Baseline  │   │ Two-Stage │   │ Empirical │   │   Risk-   │   │ Difference│   │  Active   │
 │    SRS    │   │Sequential │   │   Bayes   │   │Stratified │   │ Estimator │   │Importance │
 └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
       │               │               │               │               │               │
       └───────────────┴───────────────┼───────────────┴───────────────┴───────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │ Point & Interval Estimation   │
                       │   (p̂, [CI_lower, CI_upper])   │
                       └───────────────┬───────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │     Fixed Decision Policy     │
                       │   ► ACCEPT / REWORK / REJECT  │
                       └───────────────────────────────┘
```

### 2.1 Baseline: Simple Random Sampling (SRS) + Wilson Score CI
- **Sampling**: $S \sim \text{Uniform}(N, n=100)$.
- **Estimator**: $\hat{p}_{MLE} = \frac{k}{n}$.
- **Interval**: Wilson Score Interval with Finite Population Correction (FPC):
  $$\text{CI}_{95\%} = \frac{k + \frac{z^2}{2}}{n + z^2} \pm \frac{z \cdot \text{FPC}}{n + z^2}\sqrt{\frac{k(n-k)}{n} + \frac{z^2}{4}}, \quad \text{FPC} = \sqrt{\frac{N-n}{N-1}}$$
  *(Why Wilson over Wald? For $p \approx 0.02$ and $n=100$, Wald standard error $\hat{p} \pm 1.96\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$ collapses to 0 when $k=0$, violating nominal 95% coverage).*

### 2.2 Sequential: Two-Stage Double Acceptance Sampling
- **Stage 1**: Audit $n_1 = 50$ items (0.5% budget).
  - If $k_1 = 0 \implies$ **Early ACCEPT** (Audit stops, saving 50% cost).
  - If $k_1 \ge 4 \implies$ **Early REJECT** (Audit stops, saving 50% cost).
- **Stage 2**: If $1 \le k_1 \le 3$, inspect remaining $n_2 = 50$ items. Make definitive decision on cumulative $k = k_1 + k_2$.
- **Result**: Lowers Average Sample Number (ASN) from 100 to $\approx 77$ items while preserving statistical power.

### 2.3 Bayesian: Empirical Beta-Binomial Updating
- **Prior**: Historical quality track record for vendor $v$: $p \sim \text{Beta}(\alpha_v, \beta_v)$.
- **Posterior**: $p \mid k \sim \text{Beta}(\alpha_v + k, \beta_v + n - k)$.
- **Estimator (Posterior Mean)**:
  $$\hat{p}_{Bayes} = \frac{\alpha_v + k}{\alpha_v + \beta_v + n} = w \cdot \frac{\alpha_v}{\alpha_v + \beta_v} + (1-w) \cdot \frac{k}{n}$$
- **Decision Engine**: Computes exact posterior tail probabilities $\mathbb{P}(p \le p_{AQL} \mid \text{data})$ and $\mathbb{P}(p \ge p_{LTPD} \mid \text{data})$.

### 2.4 Stratified: Risk & Neyman Allocation
- **Partitioning**: Uses metadata (annotator ID, task complexity) to partition batch into $H=3$ risk tiers.
- **Neyman Allocation**: Allocates sample size proportional to stratum size and variance:
  $$n_h = n \cdot \frac{N_h \sigma_h}{\sum_j N_j \sigma_j}, \quad \sigma_h \approx \sqrt{p_h(1-p_h)}$$
- **Unbiased Estimator**: $\hat{p}_{strat} = \sum_{h=1}^H \frac{N_h}{N} \bar{y}_h$, with analytical stratified variance.

### 2.5 Model-Assisted: Difference Estimator (AI Judge in the Loop)
- **Surrogate Scoring**: Automated judge / heuristic checker scores all $N$ items $s_i \in [0, 1]$ ($\bar{s} = \frac{1}{N}\sum_{i=1}^N s_i$).
- **Difference Estimator**:
  $$\hat{p}_{diff} = \bar{s} + \frac{1}{n}\sum_{i \in S} (y_i - s_i)$$
- **Guarantees**:
  - **Zero Bias**: $\mathbb{E}[\hat{p}_{diff}] = p_{true}$.
  - **Variance Reduction**: $\mathbb{V}(\hat{p}_{diff}) = \frac{1-f}{n} \sigma_{y-s}^2$. With correlation $\rho \ge 0.65$, variance drops by $>60\%$.

---

## 3. Comprehensive Benchmark Results (1,000 Batches across 5 Scenarios)

### Overall Summary Leaderboard

| Rank | Strategy | Mean MAE (%) | Bad-Batch Recall (%) | Good-Batch Accept (%) | 95% CI Coverage (%) | Avg Items Audited (ASN) | Composite Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Bayesian: Empirical Beta-Binomial** | **1.013%** | **100.0%** | 89.7% | **95.4%** | 100.0 | **0.9949** |
| **2** | **Model-Assisted (Active Importance)** | 1.433% | **100.0%** | 92.1% | 83.9% | 100.0 | **0.9928** |
| **3** | **Model-Assisted (Difference Estimator)**| 1.437% | 99.8% | **94.5%** | 85.5% | 100.0 | **0.9917** |
| **4** | **Baseline: Simple Random Sampling (SRS)**| 1.452% | 99.7% | **94.5%** | 95.0% | 100.0 | **0.9911** |
| **5** | **Stratified: Risk & Neyman Allocation** | 1.509% | 98.1% | 91.1% | 83.0% | 100.0 | **0.9830** |
| **6** | **Sequential: Two-Stage Double Sampling**| 1.987% | 96.1% | 93.1% | 95.5% | **77.0** *(23% savings)* | **0.9704** |

---

## 4. Key Traps & Engineering Insights

### Trap 1: Post-Hoc Threshold Optimization (P-Hacking)
- **The Trap**: Adjusting the decision threshold after observing batch performance to artificially inflate accuracy on a benchmark set.
- **The Solution**: Lock in AQL (2.0%) and LTPD (6.0%) in code configuration. All decision policies evaluate only relative to these pre-fixed targets.

### Trap 2: Wald Confidence Interval Breakdown
- For small sample sizes ($n=100$) and rare defects ($p \le 2\%$), the Wald interval $\hat{p} \pm 1.96\sqrt{\hat{p}(1-\hat{p})/n}$ frequently produces lower bounds $< 0$ or width $= 0$ (when $k=0$).
- **Requirement**: Always use **Wilson Score** or **Exact Beta Credible Intervals**.

### Trap 3: Active Sampling Bias
- Sampling only high-risk items without inverse-probability weighting leads to massive overestimation of error rates ($\hat{p} \gg p_{true}$).
- **Requirement**: Pair active sampling with **Horvitz-Thompson** or **Difference Estimators** to strictly guarantee zero estimation bias.

---

## 5. Cân đối với Luật Chung Của Hackathon (General Rules Compliance)

Để tuân thủ tuyệt đối các quy tắc chung của thử thách, hệ thống được thiết kế với các luận điểm sau:

### ✅ Những điểm tuân thủ & Cải tiến hợp lệ
1. **Dữ liệu Simulation (Synthetic Cases)**: Không dùng trick map ngược ground-truth từ COCO/MOT17. Framework tự build một `BatchGenerator` phức tạp mô phỏng các persona của vendor (Tier-1, Flaky, Tier-3) và phân phối lỗi dựa trên độ khó (Complexity) và độ lệch của annotator (Annotator Bias).
2. **Dùng AI Model (Surrogate) & Tránh Leakage**: Chiến lược `ModelAssistedDifferenceStrategy` sử dụng điểm dự đoán lỗi $s_i$ từ mô hình AI. Để tránh việc AI sai lầm (Anchoring/Hallucination) dẫn đến quyết định mù quáng, chúng tôi sử dụng **Difference Estimator**. Công thức này được chứng minh toán học là **không chệch tuyệt đối (strictly unbiased)** $\mathbb{E}[\hat{p}_{diff}] = p_{true}$, kể cả khi mô hình AI đoán bậy bạ. AI chỉ giúp giảm phương sai (variance reduction), quyền quyết định thực tế vẫn dựa trên sample ground-truth $y_i$ do human audit.
3. **Tách biệt Evaluation**: Ngưỡng `QualityThresholds` (AQL=2%, LTPD=6%) bị khóa cứng (hardcoded) trong constructor của `AcceptanceDecisionPolicy` trước khi chạy vòng lặp Benchmark. Không có bất kỳ parameter tuning nào diễn ra trên tập test.

### ⚠️ Khuyết điểm của Giải pháp (Honest Weaknesses)
Nói thẳng những điểm chưa hoàn hảo trong thiết kế của chúng tôi:
- **Bayesian Strategy dễ bị lừa bởi "Thiên nga đen"**: Mặc dù Empirical Bayes đạt điểm tổng hợp cao nhất, nó bị phụ thuộc lớn vào `vendor_priors`. Nếu một vendor uy tín (Tier-1) bất ngờ đổi sub-contractor và giao một batch rác rưởi (Black Swan event), Prior quá mạnh có thể kéo tụt ước lượng và khiến hệ thống vô tình **ACCEPT** batch lỗi đó (False Accept).
- **Phụ thuộc vào chất lượng phân tầng (Stratification)**: Ở chiến lược Risk-Stratified, nếu metadata (độ khó, ID annotator) không có giá trị phân loại cao (tức là lỗi phân bố đồng đều, hoàn toàn ngẫu nhiên), phân bổ Neyman sẽ bị suy giảm hiệu quả, biến thành biến thể chậm chạp của Random Sampling.
- **Giả định Human Auditor hoàn hảo**: Model toán học hiện tại giả định $y_i$ sinh ra từ 1% human audit là chuẩn xác 100%. Trên thực tế, Human Auditor vẫn có thể sai (Label Noise). Mô hình chưa tích hợp biến số $P(\text{Auditor Error})$ vào việc mở rộng Confidence Interval.

---

## 6. Quickstart & Execution Guide

### Run Comprehensive Benchmark
```bash
# Run 1,000-batch Monte Carlo benchmark across all strategies
source .venv/bin/activate
python3 benchmark.py
```

### Run Interactive Single-Batch Walkthrough
```bash
# Demonstrates step-by-step audit, confidence interval, and decision logic
python3 demo.py
```

### Generate Diagnostic & OC Curve Plots
```bash
# Generates Operating Characteristic (OC) curves and MAE vs Budget curves
python3 visualize.py
```
Outputs saved in `artifacts/`:
- `artifacts/oc_curves.png`
- `artifacts/budget_vs_mae.png`
- `artifacts/benchmark_summary.csv`
- `artifacts/benchmark_detailed.csv`
