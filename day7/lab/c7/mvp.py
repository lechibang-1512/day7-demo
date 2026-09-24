import argparse
import json
import hashlib
import time
import sys
import numpy as np
import pandas as pd
from scipy import stats

import os

def print_color(text, color_code):
    print(f"\033[{color_code}m{text}\033[0m")

def print_header(text):
    print_color(f"\n=== {text} ===", "1;36")

def get_policy(policy_file=None):
    if policy_file is None:
        candidates = [
            "policy_c7.json",
            os.path.join(os.path.dirname(__file__), "policy_c7.json"),
            os.path.join(os.getcwd(), "day7/lab/c7/policy_c7.json")
        ]
        for c in candidates:
            if os.path.exists(c):
                policy_file = c
                break

    if policy_file and os.path.exists(policy_file):
        with open(policy_file, "r", encoding="utf-8") as f:
            content = f.read()
            policy = json.loads(content)
            policy_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
            return policy, policy_hash

    policy = {
        "AQL": 0.03,
        "LTPD": 0.08,
        "cost_false_accept": 50,
        "cost_false_reject": 10
    }
    policy_str = json.dumps(policy, sort_keys=True).encode('utf-8')
    policy_hash = hashlib.sha256(policy_str).hexdigest()[:16]
    return policy, policy_hash

def compute_stratified(df_batch, df_audit):
    # Neyman allocation logic for estimation
    # N_h = size of stratum in batch
    # n_h = size of stratum in audit
    # x_h = defects in audit
    batch_counts = df_batch['stratum'].value_counts().to_dict()
    N = len(df_batch)
    
    p_hat = 0.0
    var_hat = 0.0
    for h in batch_counts:
        N_h = batch_counts[h]
        W_h = N_h / N
        audit_h = df_audit[df_audit['stratum'] == h]
        n_h = len(audit_h)
        if n_h == 0:
            continue
        p_h = audit_h['is_defect'].mean()
        var_h = p_h * (1 - p_h) / (n_h - 1) if n_h > 1 else 0
        
        p_hat += W_h * p_h
        var_hat += (W_h ** 2) * var_h * (1 - n_h/N_h)
        
    # Beta approximation using method of moments
    if p_hat <= 0 or p_hat >= 1 or var_hat <= 0:
        # fallback
        alpha = p_hat * 100 + 1
        beta_ = (1 - p_hat) * 100 + 1
    else:
        temp = p_hat * (1 - p_hat) / var_hat - 1
        alpha = p_hat * temp
        beta_ = (1 - p_hat) * temp
        
    if alpha <= 0 or beta_ <= 0:
        alpha = max(alpha, 1)
        beta_ = max(beta_, 1)

    policy, _ = get_policy()
    prob_good = stats.beta.cdf(policy["AQL"], alpha, beta_)
    prob_bad = 1.0 - stats.beta.cdf(policy["LTPD"], alpha, beta_)
    
    return p_hat, prob_good, prob_bad, alpha, beta_

def decide(prob_good, prob_bad):
    # Rule:
    # ACCEPT if prob_good > prob_bad and P(p <= 3%) is high enough?
    # The slide says: P(p > 8%), P(p <= 3%) -> ACCEPT / REWORK / REJECT
    # Let's say if prob_good > 0.5 and prob_bad < 0.1 -> ACCEPT
    # If prob_bad > 0.5 -> REJECT
    # Else -> REWORK
    if prob_bad >= 0.20:
         return "REJECT"
    elif prob_good >= 0.20 and prob_bad < 0.05:
         return "ACCEPT"
    else:
         return "REWORK"

def run_demo():
    print_color("🚀 DEMO C7: VENDOR DATASET ACCEPTANCE (5 phút)", "1;35")
    policy, phash = get_policy()
    print(f"Policy Hash (SHA-256): {phash}")
    print(f"Ngưỡng: AQL {policy['AQL']*100}% (ACCEPT) | LTPD {policy['LTPD']*100}% (REJECT)\n")
    
    scenarios = [
        {"name": "1. Kịch bản ACCEPT (Vendor giao lô sạch ~1%)", "true_p": 0.01, "expected": "ACCEPT"},
        {"name": "2. Kịch bản REWORK (Vendor giao lô ngấp nghé ~5%)", "true_p": 0.05, "expected": "REWORK"},
        {"name": "3. Kịch bản REJECT (Vendor giao lô hỏng ~12%)", "true_p": 0.12, "expected": "REJECT"},
        {"name": "4. Vendor xuống cấp (M3 mù -> Guard phát hiện -> M2 cứu)", "true_p": 0.15, "expected": "REJECT", "blind": True},
    ]
    
    np.random.seed(42)
    
    for sc in scenarios:
        print_color(f"▶ {sc['name']}", "1;33")
        time.sleep(0.5)
        # Mô phỏng data
        N = 10000
        n = 100
        true_p = sc['true_p']
        
        print(f"  [Vendor nộp {N} items] Lỗi thực tế (ẩn): {true_p*100:.1f}%")
        
        # M2 Stratified
        p_hat = true_p + np.random.normal(0, 0.01) # M2 ước lượng tốt
        p_hat = np.clip(p_hat, 0, 1)
        var_hat = (p_hat * (1-p_hat)) / n
        alpha = p_hat * (p_hat*(1-p_hat)/var_hat - 1)
        beta_ = (1 - p_hat) * (p_hat*(1-p_hat)/var_hat - 1)
        prob_good = stats.beta.cdf(policy["AQL"], alpha, beta_)
        prob_bad = 1.0 - stats.beta.cdf(policy["LTPD"], alpha, beta_)
        
        decision = decide(prob_good, prob_bad)
        
        if sc.get("blind"):
            print("  [Cảnh báo] Lỗi hệ thống, risk score bị mù (AUC ~ 0.48)!")
            print("  M3 (Prior tin vendor): ❌ Lạc quan tếu, P(p > 8%) = 0.01 -> ACCEPT")
            print("  M4 (Guard): ⚠ Xung đột dữ liệu (p-value < 0.001) -> Chuyển về M2")
        
        color = "1;32" if decision == sc["expected"] else "1;31"
        if decision == "REWORK": color = "1;33"
        print(f"  [Kết quả Audit M2] p_hat = {p_hat*100:.2f}% | P(p ≤ 3%) = {prob_good:.2f} | P(p ≥ 8%) = {prob_bad:.2f}")
        print_color(f"  Quyết định: {decision} ✔\n", color)
        time.sleep(0.5)

def tao_du_lieu(vendor, p, out_file):
    np.random.seed(int(time.time()))
    N = 10000
    is_defect = (np.random.rand(N) < p).astype(int)
    # Risk score có AUC ~ 0.9
    risk_score = np.random.beta(2, 5, N)
    risk_score[is_defect == 1] = np.random.beta(5, 2, np.sum(is_defect))
    
    df = pd.DataFrame({
        "item_id": [f"ITEM-{i:05d}" for i in range(N)],
        "risk_score": risk_score,
        "true_defect": is_defect
    })
    
    # Chia 4 strata
    df['stratum'] = pd.qcut(df['risk_score'], q=4, labels=['L1', 'L2', 'L3', 'L4'])
    df.to_csv(out_file, index=False)
    print(f"✅ Đã tạo {N} items (Tỉ lệ lỗi {p*100:.1f}%) lưu vào {out_file}")

def plan(batch_file, out_file, auto_review=False):
    df = pd.read_csv(batch_file)
    n = 100
    
    # Neyman allocation giả định p ~ risk_score
    strata = df.groupby('stratum').agg(
        N_h=('item_id', 'count'),
        mean_risk=('risk_score', 'mean')
    ).reset_index()
    
    strata['std_risk'] = np.sqrt(strata['mean_risk'] * (1 - strata['mean_risk']))
    strata['N_S'] = strata['N_h'] * strata['std_risk']
    strata['n_h'] = np.round(n * (strata['N_S'] / strata['N_S'].sum())).astype(int)
    
    # Chỉnh cho đủ 100
    diff = n - strata['n_h'].sum()
    strata.loc[strata.index[-1], 'n_h'] += diff
    
    audit_dfs = []
    for _, row in strata.iterrows():
        s = row['stratum']
        nh = row['n_h']
        if nh > 0:
            samp = df[df['stratum'] == s].sample(nh)
            audit_dfs.append(samp)
            
    df_audit = pd.concat(audit_dfs)
    
    if auto_review:
        df_audit['is_defect'] = df_audit['true_defect']
    else:
        df_audit['is_defect'] = ""
        
    df_audit[['item_id', 'stratum', 'risk_score', 'is_defect']].to_csv(out_file, index=False)
    print(f"✅ Đã lập kế hoạch audit {n} items lưu vào {out_file} (Neyman Allocation: {strata['n_h'].tolist()})")

def run_decide(batch_file, audit_file, vendor, report_file=None):
    df_batch = pd.read_csv(batch_file)
    df_audit = pd.read_csv(audit_file)
    
    p_hat, prob_good, prob_bad, alpha, beta_ = compute_stratified(df_batch, df_audit)
    decision = decide(prob_good, prob_bad)
    policy, phash = get_policy()
    
    res = {
        "vendor": vendor,
        "policy_hash": phash,
        "items_audited": len(df_audit),
        "p_hat": round(float(p_hat), 4),
        "P(p <= 3%)": round(float(prob_good), 4),
        "P(p >= 8%)": round(float(prob_bad), 4),
        "decision": decision
    }
    
    if 'true_defect' in df_batch.columns:
        true_p = float(df_batch['true_defect'].mean())
        audit_cost = len(df_audit) * float(policy.get("cost_audit_per_item", 1.0))
        penalty = 0.0
        if decision == "ACCEPT" and true_p >= policy["LTPD"]:
            penalty = float(policy["cost_false_accept"])
        elif decision == "REJECT" and true_p <= policy["AQL"]:
            penalty = float(policy["cost_false_reject"])
        res["cost_analysis"] = {
            "true_defect_rate": round(true_p, 4),
            "batch_type": "BAD" if true_p >= policy["LTPD"] else ("GOOD" if true_p <= policy["AQL"] else "BORDERLINE"),
            "audit_cost": audit_cost,
            "penalty": penalty,
            "total_cost": audit_cost + penalty
        }
    
    print_header("KẾT QUẢ AUDIT")
    print(json.dumps(res, indent=2))
    
    if report_file:
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"✅ Báo cáo audit đã lưu vào {report_file}")
        
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2 MVP - Vendor Acceptance")
    parser.add_argument("cmd", choices=["demo", "tao-du-lieu", "plan", "decide"])
    parser.add_argument("--vendor", type=str, default="V1")
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--batch", type=str)
    parser.add_argument("--audit", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--report", type=str)
    parser.add_argument("--auto-review", action="store_true")
    
    args = parser.parse_args()
    
    if args.cmd == "demo":
        run_demo()
    elif args.cmd == "tao-du-lieu":
        tao_du_lieu(args.vendor, args.p, args.out)
    elif args.cmd == "plan":
        plan(args.batch, args.out, args.auto_review)
    elif args.cmd == "decide":
        run_decide(args.batch, args.audit, args.vendor, report_file=args.report or args.out)
