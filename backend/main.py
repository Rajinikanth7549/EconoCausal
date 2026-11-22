from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io

from causal_analysis import run_causal_analysis
from budget_optimizer import optimize_budget

app = FastAPI(title="EconoCausal API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "EconoCausal API is running"}

@app.post("/analyze/full")
async def analyze_full_pipeline(
    file: UploadFile = File(...),
    budget: float = Form(50000.0)
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        purchase_rate = round(float(df["purchased"].mean() * 100), 2) if "purchased" in df.columns else 0.0
        avg_order = round(float(df["list_price"].mean()), 2) if "list_price" in df.columns else 0.0
        summary = {
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "purchase_rate_percent": purchase_rate,
            "avg_list_price": avg_order,
        }

        causal = run_causal_analysis(df)

        ite_lookup = {row["customer_id"]: row["estimated_ite"] for row in causal["customers"] if "customer_id" in row}
        if "customer_id" in df.columns and len(ite_lookup) > 0:
            ite_series = df["customer_id"].map(ite_lookup).fillna(
                df["true_treatment_effect"] if "true_treatment_effect" in df.columns else causal["estimated_ate"]
            ).values
        elif "true_treatment_effect" in df.columns:
            ite_series = df["true_treatment_effect"].values
        else:
            ite_series = np.full(len(df), causal["estimated_ate"])

        order_values = df["list_price"].values if "list_price" in df.columns else np.full(len(df), 100.0)
        discount_costs = df["discount_cost"].values if "discount_cost" in df.columns else np.full(len(df), 10.0)

        opt = optimize_budget(ite_series, order_values, discount_costs, budget)
        selected_set = set(opt["selected_customer_ids"])

        expected_rev = float(np.sum((ite_series * order_values)[opt["selected_customer_ids"]])) if opt["selected_customer_ids"] else 0.0
        blanket_cost = float(np.sum(discount_costs))
        savings = max(0.0, blanket_cost - opt["total_budget_spent"])
        roi_mult = round(expected_rev / opt["total_budget_spent"], 2) if opt["total_budget_spent"] > 0 else 0.0

        allocation_rows = []
        for idx in range(min(250, len(df))):
            row = df.iloc[idx]
            targeted = idx in selected_set
            ite_v = float(ite_series[idx])
            lp_v = float(order_values[idx])
            dc_v = float(discount_costs[idx])
            allocation_rows.append({
                "customer_id": str(row["customer_id"]) if "customer_id" in df.columns else f"C{idx+1:05d}",
                "region": str(row["region"]) if "region" in df.columns else "N/A",
                "customer_segment": str(row["customer_segment"]) if "customer_segment" in df.columns else "Standard",
                "estimated_ite": round(ite_v, 4),
                "list_price": round(lp_v, 2),
                "discount_cost": round(dc_v, 2),
                "expected_net_profit": round((ite_v * lp_v) - dc_v, 2),
                "action": "Assign Coupon" if targeted else "Holdout ($0)",
                "targeted": targeted,
            })

        allocation_rows.sort(key=lambda r: (not r["targeted"], -r["expected_net_profit"]))

        roi_summary = (
            f"SciPy MILP allocated ${opt['total_budget_spent']:,.2f} of your ${budget:,.2f} budget to "
            f"{opt['selected_customer_count']:,} high-uplift Persuadables, generating "
            f"${expected_rev:,.2f} in expected incremental revenue (${opt['expected_incremental_profit']:,.2f} net profit, "
            f"{roi_mult}x ROI) and saving ${savings:,.2f} compared to blanket discounting."
        )

        return {
            "summary": summary,
            "causal": causal,
            "optimization": {
                "budget_cap": round(budget, 2),
                "selected_customer_count": opt["selected_customer_count"],
                "total_budget_spent": opt["total_budget_spent"],
                "budget_utilization_percent": round((opt["total_budget_spent"] / budget) * 100, 1) if budget > 0 else 0.0,
                "expected_incremental_revenue": round(expected_rev, 2),
                "expected_incremental_profit": opt["expected_incremental_profit"],
                "roi_multiplier": roi_mult,
                "savings_vs_blanket": round(savings, 2),
                "roi_summary": roi_summary,
                "allocation_matrix": allocation_rows,
            },
        }
    except Exception as e:
        print(f"ERROR IN FULL PIPELINE: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")
